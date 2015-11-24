import os
import pwd
import sys

from lxml import etree
from sqlalchemy import create_engine, Column, Integer, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects import postgresql

# Create database for storing the taxonomy
user = pwd.getpwuid(os.getuid()).pw_name
os.system('psql -c "CREATE DATABASE sharetaxonomy;"')
engine = create_engine('postgresql://{}@localhost/sharetaxonomy'.format(user), echo=True)
Session = sessionmaker(bind=engine)


# define models
Base = declarative_base()


class Term(Base):
    __tablename__ = 'terms'
    term_id = Column(Integer, primary_key=True)
    term = Column(Text)


class BroaderTerm(Base):
    __tablename__ = 'broaderterms'
    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.term_id'))
    broader_term = Column(Text)


class NarrowerTerm(Base):
    __tablename__ = 'narrowerterms'
    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.term_id'))
    narrower_term = Column(Text)


class RelatedTerm(Base):
    __tablename__ = 'relatedterms'
    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.term_id'))
    related_term = Column(Text)


class Synonym(Base):
    __tablename__ = 'synonyms'
    id = Column(Integer, primary_key=True)
    term_id = Column(Integer, ForeignKey('terms.term_id'))
    synonym = Column(Text)


class Node(Base):
    __tablename__ = 'taxonomytree'
    parent_id = Column(Integer, ForeignKey('terms.term_id'), primary_key=True)
    parent_term = Column(Text)
    child_id = Column(Integer, ForeignKey('terms.term_id'), primary_key=True)
    child_term = Column(Text)
    path = Column(postgresql.ARRAY(Integer), primary_key=True)
    depth = Column(Integer)


# Define the actions
def inject():
    try:
        thesaurus = etree.parse('plosthes.2015-2.extract.xml')
    except:
        try:
            thesaurus = etree.parse('../plosthes.2015-2.extract.xml')
        except:
            try:
                thesaurus = etree.parse('../../plosthes.2015-2.extract.xml')
            except IOError:
                raise

    terms = thesaurus.xpath('TermInfo')
    session = Session()
    for i, term in enumerate(terms):
        term_id = i + 1
        session.add(
            Term(term_id=term_id, term=term.xpath('T/node()')[0])
        )
    session.commit()
    for i, term in enumerate(terms):
        term_id = i + 1
        for bt in term.xpath('BT/node()'):
            session.add(
                BroaderTerm(
                    term_id=term_id,
                    broader_term=bt
                )
            )
        for nt in term.xpath('NT/node()'):
            session.add(
                NarrowerTerm(
                    term_id=term_id,
                    narrower_term=nt
                )
            )
        for rt in term.xpath('BT/node()'):
            session.add(
                RelatedTerm(
                    term_id=term_id,
                    related_term=rt
                )
            )
        for synonym in term.xpath('Synonym/node()'):
            session.add(
                Synonym(
                    term_id=term_id,
                    synonym=synonym
                )
            )
    session.commit()


def polish():
    """
    add term id for broader terms, narrower terms, related terms
    """
    session = Session()
    session.execute(
        '''
        ALTER TABLE broaderterms ADD COLUMN bt_id Integer REFERENCES terms(term_id);
        UPDATE broaderterms SET bt_id=terms.term_id FROM terms WHERE broader_term=terms.term;
        ALTER TABLE narrowerterms ADD COLUMN nt_id Integer REFERENCES terms(term_id);
        UPDATE narrowerterms SET nt_id=terms.term_id FROM terms WHERE narrower_term=terms.term;
        ALTER TABLE relatedterms ADD COLUMN rt_id Integer REFERENCES terms(term_id);
        UPDATE relatedterms SET rt_id=terms.term_id FROM terms WHERE related_term=terms.term;
        '''
    )
    session.commit()


# Create tables
def initialize():
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)


def get_roots():
    session = Session()
    session.execute(
        '''
        SELECT * INTO TABLE roots FROM terms t WHERE NOT EXISTS (SELECT 1 FROM broaderterms b WHERE b.term_id = t.term_id);
        '''
    )
    session.commit()


def get_children(root_id):
    """
    Where amazing happens!
    This method gets the children, paths, and depth given any term_id as your root.
    """
    session = Session()
    query_string = '''
        WITH RECURSIVE nodes(parent_id, parent_term, child_id, child_term, path, depth) AS (
            SELECT
                n."term_id", t1."term",
                n."nt_id", t2."term",
                ARRAY[n."term_id"], 1
            FROM "narrowerterms" AS n, "terms" AS t1, "terms" AS t2
            WHERE n."term_id" = {} -- root id
            AND t1."term_id" = n."term_id" AND t2."term_id" = n."nt_id"
            UNION ALL
            SELECT
                n."term_id", t1."term",
                n."nt_id", t2."term",
                path || n."term_id", nd.depth + 1
            FROM "narrowerterms" AS n, "terms" AS t1, "terms" AS t2,
                nodes AS nd
            WHERE n."term_id" = nd.child_id
            AND t1."term_id" = n."term_id" AND t2."term_id" = n."nt_id" -- AND depth < your limit
        )
        INSERT INTO taxonomytree (SELECT * FROM nodes);
        '''.format(root_id)
    session.execute(query_string)
    session.commit()


def grow_tree():
    Base.metadata.create_all(engine)
    roots = engine.execute('SELECT term_id FROM roots')
    for root in roots:
        root_id = int(root[0])
        get_children(root_id)


# Execute
def construct_taxonomy():
    initialize()
    inject()
    polish()
    grow_tree()


if __name__ == '__main__':
    construct_taxonomy()

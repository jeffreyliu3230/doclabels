import doclabels as dl

docs = dl.read(filename)

labels = dl.WMD(docs, taxonomy='sharetaxonomy')

# Or
labels = dl.MAI(docs, taxonomy='sharetaxonomy')

# Or
labels = dl.CNN(data=docs, taxonomy='sharetaxonomy')

# Or
labels = dl.ensemble(data=docs, method=['MAI', 'WMD', 'CNN'], taxonomy='sharetaxonomy')

labels.result

labels.summary

# Evaluation
result = dl.eval(labels, true_labels)


# Algorithms
# WMD
from gensim.models import Word2Vec


def WMD(docs, taxonomy='sharetaxonomy'):
    terms = docs.extract_term()
    # do stuff
    engine = create_engine('postgresql://{}@localhost/{}'.format(user, taxonomy), echo=True)
    terms = engine.query('SELECT * from terms')
    doc_terms = term_extract(docs)
    for doc_term


    return labels


#include "BronKerbosch.h"
#include <vector>
#include <stdexcept>
#include "opencv2/opencv.hpp" // OpenCV 2.4.5

using namespace std;

class Vertex;

class VertexSet
{
	public:
		// Constructor
		VertexSet(int totalVertices);
		VertexSet(int totalVertices, bool initVal);
		VertexSet(const VertexSet& other);
		const VertexSet& operator=(const VertexSet& rhs);
		
		bool isEmpty() const;
		Vertex getAnyVertex() const;
		VertexSet unionWith(const Vertex& v) const;
		VertexSet intersectWith(const VertexSet& vs) const;
		void remove(const Vertex& v);
		vector<int> getVertexIndices() const;

		static vector<Vertex> graph;

	private:
		void copy(const VertexSet& other);
		int nVertices;
		vector<bool> hasVertices;
};


class Vertex
{
	public:
		Vertex(int id, int totalVertices);
		Vertex(const Vertex& v);
		const Vertex& operator=(const Vertex& rhs);
		
		int id;
		VertexSet neighbors;
};





Vertex::Vertex(int i, int totalVertices)
	:id(i), neighbors(totalVertices, false)
{
	// Empty.
}

Vertex::Vertex(const Vertex& v)
	:id(v.id), neighbors(v.neighbors)
{
	// Empty.
}

const Vertex& Vertex::operator=(const Vertex& rhs)
{
	this->id = rhs.id;
	this->neighbors = rhs.neighbors;
	return *this;
}


//


vector<Vertex> VertexSet::graph;


VertexSet::VertexSet(int totalVertices, bool initVal)
{
	this->nVertices = (initVal ? totalVertices : 0);
	this->hasVertices = vector<bool>(totalVertices, initVal);
}

VertexSet::VertexSet(const VertexSet& other)
{
	copy(other);
}

const VertexSet& VertexSet::operator=(const VertexSet& rhs)
{
	copy(rhs);
	return *this;
}

void VertexSet::copy(const VertexSet& other)
{
	this->nVertices = other.nVertices;
	this->hasVertices = other.hasVertices;
}

bool VertexSet::isEmpty() const
{
	return (this->nVertices <= 0);
}

Vertex VertexSet::getAnyVertex() const
{
	int totalVertices = this->hasVertices.size();
	for (int i = 0; i < totalVertices; ++i)
	{
		if (this->hasVertices[i])
		{
			return graph[i];
		}
	}
	
	throw runtime_error("Tried to get vertex out of empty vertex set.");
}

VertexSet VertexSet::unionWith(const Vertex& v) const
{
	VertexSet out(*this);
	
	if (!out.hasVertices[v.id])
	{
		out.hasVertices[v.id] = true;
		++ out.nVertices;
	}
	
	return out;
}

VertexSet VertexSet::intersectWith(const VertexSet& vs) const
{
	int totalVertices = this->hasVertices.size();
	VertexSet out(*this);

	for (int i = 0; i < totalVertices; ++i)
	{
		if (out.hasVertices[i] && !vs.hasVertices[i])
		{
			out.hasVertices[i] = false;
			-- out.nVertices;
		}
	}

	return out;
}

void VertexSet::remove(const Vertex& v)
{
	if (this->hasVertices[v.id])
	{
		this->hasVertices[v.id] = false;
		-- this->nVertices;
	}
}

vector<int> VertexSet::getVertexIndices() const
{
	vector<int> v;
	int totalVertices = this->hasVertices.size();

	v.reserve(this->nVertices);
	for (int i = 0; i < totalVertices; ++i)
	{
		if (this->hasVertices[i]) v.push_back(i);
	}

	return v;
}

/*
BronKerbosch(R, P, X):
	if P and X are both empty:
		report R as a maximal clique
	for each vertex v in P:
		BronKerbosch1(R ⋃ {v}, P ⋂ N(v), X ⋂ N(v))
		P := P \ {v}
		X := X ⋃ {v}
*/
static vector< VertexSet > cliques;

static void bronKerbosch(VertexSet r, VertexSet p, VertexSet x)
{
	if (p.isEmpty() && x.isEmpty())
	{
		cout << cliques.size() << "/" << cliques.capacity() << endl;
		cliques.push_back(r);
		return;
	}

	while (!p.isEmpty())
	{
		Vertex v = p.getAnyVertex();
		bronKerbosch(r.unionWith(v), p.intersectWith(v.neighbors), x.intersectWith(v.neighbors));
		p.remove(v);
		x = x.unionWith(v);
	}
}


void findMaximalCliques(const cv::SparseMat& adjMat, vector< vector<int> >& groups)
{
	int nVertices = adjMat.size()[0];
	vector<Vertex> graph;
	
	graph.reserve(nVertices);
	for (int i = 0; i < nVertices; ++i)
	{
		graph.push_back(Vertex(i, nVertices));
	}

	{
		cv::SparseMatConstIterator_<uchar> it = adjMat.begin<uchar>();
		cv::SparseMatConstIterator_<uchar> itEnd = adjMat.end<uchar>();

		for (; it != itEnd; ++it)
		{
			const cv::SparseMat::Node* node = it.node();
			int i = node->idx[0]; // Row.
			int j = node->idx[1]; // Column.

			Vertex vi(i, nVertices);
			Vertex vj(j, nVertices);

			graph[i].neighbors = graph[i].neighbors.unionWith(vj);
			graph[j].neighbors = graph[j].neighbors.unionWith(vi);
		}
	}

	VertexSet::graph = graph;
	VertexSet r(nVertices, false);
	VertexSet p(nVertices, true);
	VertexSet x(nVertices, false);
	cliques.reserve(nVertices);
	bronKerbosch(r, p, x);
	
	int nGroups = cliques.size();
	groups.resize(nGroups);
	for (int i = 0; i < nGroups; ++i)
	{
		groups[i] = cliques[i].getVertexIndices();
	}
}

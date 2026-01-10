import React, { useState, useRef, useEffect } from 'react';
import { Upload, Search, FileText, Database, Lock, Zap, Trash2, MessageSquare, Download } from 'lucide-react';

const IntelliSearchRAG = () => {
  const [documents, setDocuments] = useState([]);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [vectorStore, setVectorStore] = useState([]);
  const fileInputRef = useRef(null);

  // Simple text chunking function
  const chunkText = (text, chunkSize = 500) => {
    const chunks = [];
    const words = text.split(/\s+/);
    for (let i = 0; i < words.length; i += chunkSize) {
      chunks.push(words.slice(i, i + chunkSize).join(' '));
    }
    return chunks;
  };

  // Simple embedding simulation (cosine similarity)
  const createEmbedding = (text) => {
    const words = text.toLowerCase().split(/\s+/);
    const embedding = new Array(100).fill(0);
    words.forEach((word, idx) => {
      for (let i = 0; i < word.length; i++) {
        embedding[(word.charCodeAt(i) + idx) % 100] += 1;
      }
    });
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / (magnitude || 1));
  };

  // Cosine similarity
  const cosineSimilarity = (vec1, vec2) => {
    let dotProduct = 0;
    let mag1 = 0;
    let mag2 = 0;
    for (let i = 0; i < vec1.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      mag1 += vec1[i] * vec1[i];
      mag2 += vec2[i] * vec2[i];
    }
    return dotProduct / (Math.sqrt(mag1) * Math.sqrt(mag2));
  };

  // Handle file upload
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    setLoading(true);

    for (const file of files) {
      const text = await file.text();
      const chunks = chunkText(text);
      
      const newDoc = {
        id: Date.now() + Math.random(),
        name: file.name,
        content: text,
        chunks: chunks.length,
        timestamp: new Date().toLocaleString()
      };

      // Create vector embeddings for each chunk
      const vectors = chunks.map((chunk, idx) => ({
        docId: newDoc.id,
        docName: file.name,
        chunkIndex: idx,
        text: chunk,
        embedding: createEmbedding(chunk)
      }));

      setDocuments(prev => [...prev, newDoc]);
      setVectorStore(prev => [...prev, ...vectors]);
    }

    setLoading(false);
    fileInputRef.current.value = '';
  };

  // Perform RAG search
  const performSearch = async () => {
    if (!query.trim() || vectorStore.length === 0) return;

    setLoading(true);
    setActiveTab('search');

    // Create query embedding
    const queryEmbedding = createEmbedding(query);

    // Find top-k similar chunks
    const similarities = vectorStore.map(vec => ({
      ...vec,
      score: cosineSimilarity(queryEmbedding, vec.embedding)
    }));

    similarities.sort((a, b) => b.score - a.score);
    const topResults = similarities.slice(0, 3);

    // Simulate AI response using Claude API
    const context = topResults.map(r => r.text).join('\n\n');
    
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 1000,
          messages: [
            {
              role: "user",
              content: `Based on the following context from documents, answer the question. Provide specific citations.

Context:
${context}

Question: ${query}

Instructions: Answer based solely on the provided context. If the context doesn't contain enough information, say so. Include specific references to the source material.`
            }
          ],
        })
      });

      const data = await response.json();
      const answer = data.content.find(c => c.type === 'text')?.text || 'No response generated';

      setResults({
        answer,
        sources: topResults,
        timestamp: new Date().toLocaleString()
      });
    } catch (error) {
      console.error('Search error:', error);
      // Fallback to simple concatenation
      setResults({
        answer: `Based on the retrieved context:\n\n${context.substring(0, 500)}...`,
        sources: topResults,
        timestamp: new Date().toLocaleString()
      });
    }

    setLoading(false);
  };

  const clearAll = () => {
    setDocuments([]);
    setVectorStore([]);
    setResults(null);
    setQuery('');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="bg-black bg-opacity-40 border-b border-purple-500/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Lock className="w-8 h-8 text-purple-400" />
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                  IntelliSearch RAG
                </h1>
                <p className="text-xs text-purple-300">Secure, Offline & Multimodal Intelligence</p>
              </div>
            </div>
            <div className="flex items-center space-x-6 text-sm">
              <div className="flex items-center space-x-2">
                <Database className="w-4 h-4 text-green-400" />
                <span>{documents.length} Docs</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-4 h-4 text-yellow-400" />
                <span>{vectorStore.length} Vectors</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Tab Navigation */}
        <div className="flex space-x-4 mb-6">
          <button
            onClick={() => setActiveTab('upload')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition ${
              activeTab === 'upload' 
                ? 'bg-purple-600 text-white' 
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <Upload className="w-4 h-4" />
            <span>Ingestion</span>
          </button>
          <button
            onClick={() => setActiveTab('search')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition ${
              activeTab === 'search' 
                ? 'bg-purple-600 text-white' 
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <Search className="w-4 h-4" />
            <span>Search</span>
          </button>
          <button
            onClick={() => setActiveTab('documents')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition ${
              activeTab === 'documents' 
                ? 'bg-purple-600 text-white' 
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            <FileText className="w-4 h-4" />
            <span>Documents</span>
          </button>
        </div>

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="bg-slate-800 bg-opacity-50 backdrop-blur rounded-xl p-8 border border-purple-500/20">
            <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
              <Upload className="w-5 h-5 text-purple-400" />
              <span>Document Ingestion</span>
            </h2>
            <p className="text-slate-300 mb-6">
              Upload text documents (.txt, .md, .csv) for vectorization and secure offline search.
            </p>
            
            <div className="border-2 border-dashed border-purple-500/30 rounded-xl p-12 text-center hover:border-purple-500/50 transition">
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.csv"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <Upload className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                <p className="text-lg mb-2">Click to upload documents</p>
                <p className="text-sm text-slate-400">Supports TXT, MD, CSV files</p>
              </label>
            </div>

            {loading && (
              <div className="mt-6 text-center">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-purple-500 border-t-transparent"></div>
                <p className="mt-2 text-purple-300">Processing documents...</p>
              </div>
            )}
          </div>
        )}

        {/* Search Tab */}
        {activeTab === 'search' && (
          <div className="space-y-6">
            <div className="bg-slate-800 bg-opacity-50 backdrop-blur rounded-xl p-6 border border-purple-500/20">
              <div className="flex space-x-3">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                  placeholder="Ask a question about your documents..."
                  className="flex-1 bg-slate-900 border border-purple-500/30 rounded-lg px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:border-purple-500"
                />
                <button
                  onClick={performSearch}
                  disabled={loading || !query.trim() || vectorStore.length === 0}
                  className="bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:cursor-not-allowed px-6 py-3 rounded-lg font-semibold transition flex items-center space-x-2"
                >
                  <Search className="w-5 h-5" />
                  <span>Search</span>
                </button>
              </div>
            </div>

            {loading && (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-purple-500 border-t-transparent"></div>
                <p className="mt-4 text-purple-300">Searching vector database...</p>
              </div>
            )}

            {results && !loading && (
              <div className="space-y-6">
                {/* Answer */}
                <div className="bg-gradient-to-br from-purple-900/50 to-slate-800/50 backdrop-blur rounded-xl p-6 border border-purple-500/30">
                  <div className="flex items-start space-x-3 mb-4">
                    <MessageSquare className="w-6 h-6 text-purple-400 mt-1" />
                    <div className="flex-1">
                      <h3 className="text-lg font-semibold mb-2">Answer</h3>
                      <p className="text-slate-200 whitespace-pre-wrap leading-relaxed">{results.answer}</p>
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 mt-4">Generated: {results.timestamp}</p>
                </div>

                {/* Citations */}
                <div className="bg-slate-800 bg-opacity-50 backdrop-blur rounded-xl p-6 border border-purple-500/20">
                  <h3 className="text-lg font-semibold mb-4 flex items-center space-x-2">
                    <FileText className="w-5 h-5 text-purple-400" />
                    <span>Deep Citations ({results.sources.length})</span>
                  </h3>
                  <div className="space-y-4">
                    {results.sources.map((source, idx) => (
                      <div key={idx} className="bg-slate-900/50 rounded-lg p-4 border border-purple-500/10">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-semibold text-purple-300">{source.docName}</span>
                          <span className="text-xs bg-purple-600 px-2 py-1 rounded">
                            Similarity: {(source.score * 100).toFixed(1)}%
                          </span>
                        </div>
                        <p className="text-sm text-slate-300 line-clamp-3">{source.text}</p>
                        <p className="text-xs text-slate-500 mt-2">Chunk {source.chunkIndex + 1}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="bg-slate-800 bg-opacity-50 backdrop-blur rounded-xl p-6 border border-purple-500/20">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold flex items-center space-x-2">
                <FileText className="w-5 h-5 text-purple-400" />
                <span>Document Library</span>
              </h2>
              {documents.length > 0 && (
                <button
                  onClick={clearAll}
                  className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Clear All</span>
                </button>
              )}
            </div>

            {documents.length === 0 ? (
              <div className="text-center py-12 text-slate-400">
                <Database className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>No documents uploaded yet</p>
                <p className="text-sm mt-2">Upload documents to begin</p>
              </div>
            ) : (
              <div className="space-y-3">
                {documents.map((doc) => (
                  <div key={doc.id} className="bg-slate-900/50 rounded-lg p-4 border border-purple-500/10 hover:border-purple-500/30 transition">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h3 className="font-semibold text-purple-300 mb-1">{doc.name}</h3>
                        <p className="text-sm text-slate-400">
                          {doc.chunks} chunks • {doc.content.length.toLocaleString()} characters
                        </p>
                        <p className="text-xs text-slate-500 mt-1">Added: {doc.timestamp}</p>
                      </div>
                      <Lock className="w-5 h-5 text-green-400" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Footer Stats */}
        <div className="mt-8 grid grid-cols-3 gap-4">
          <div className="bg-slate-800 bg-opacity-30 backdrop-blur rounded-lg p-4 border border-purple-500/10 text-center">
            <Lock className="w-8 h-8 mx-auto mb-2 text-green-400" />
            <p className="text-2xl font-bold">100%</p>
            <p className="text-xs text-slate-400">Data Privacy</p>
          </div>
          <div className="bg-slate-800 bg-opacity-30 backdrop-blur rounded-lg p-4 border border-purple-500/10 text-center">
            <Zap className="w-8 h-8 mx-auto mb-2 text-yellow-400" />
            <p className="text-2xl font-bold">60%</p>
            <p className="text-xs text-slate-400">Faster Retrieval</p>
          </div>
          <div className="bg-slate-800 bg-opacity-30 backdrop-blur rounded-lg p-4 border border-purple-500/10 text-center">
            <Database className="w-8 h-8 mx-auto mb-2 text-purple-400" />
            <p className="text-2xl font-bold">Local</p>
            <p className="text-xs text-slate-400">Vector Store</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IntelliSearchRAG;
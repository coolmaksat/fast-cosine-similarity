/*
 * Licensed to Elasticsearch under one or more contributor
 * license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright
 * ownership. Elasticsearch licenses this file to you under
 * the Apache License, Version 2.0 (the "License"); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.staysense.fastcosinesimilarity;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.Map;
import java.util.ArrayList;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

import com.elasticsearch.staysense.fastcosinesimilarity.Util;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.store.ByteArrayDataInput;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.plugins.Plugin;
import org.elasticsearch.plugins.ScriptPlugin;
import org.elasticsearch.script.ScriptContext;
import org.elasticsearch.script.ScriptEngine;
import org.elasticsearch.script.ScoreScript;


public final class FastCosineSimilarityPlugin extends Plugin implements ScriptPlugin{

    @Override
    public ScriptEngine getScriptEngine(Settings settings, Collection<ScriptContext<?>> contexts) {
        return new FastCosineSimilarityEngine();
    }

    private static class FastCosineSimilarityEngine implements ScriptEngine {

	/**
	 * The normalized vector score from the query
	 */
	
	double queryVectorNorm;

	@Override
	public String getType() {
	    return "fast_cosine";
	}

	@Override
	public <T> T compile(String scriptName, String scriptSource, ScriptContext<T> context, Map<String, String> params) {
	    if (context.equals(ScoreScript.CONTEXT) == false) {
		throw new IllegalArgumentException(getType() + " scripts cannot be used for context [" + context.name + "]");
	    }
	    // we use the script "source" as the script identifier
	    if ("staysense".equals(scriptSource)) {
		ScoreScript.Factory factory = (p, lookup) -> new ScoreScript.LeafFactory() {
		    // The field to compare against
		    final String field;
		    //Whether this search should be cosine or dot product
		    final Boolean cosine;
		    //The query embedded vector
		    final Object vector;
		    Boolean exclude;
		    //The final comma delimited vector representation of the query vector
		    double[] inputVector;
		    {
			if (p.containsKey("field") == false) {
			    throw new IllegalArgumentException("Missing parameter [field]");
			}

			//Determine if cosine
			final Object cosineBool = p.get("cosine");
			cosine = cosineBool != null ? (boolean)cosineBool : true;

			//Get the field value from the query
			field = p.get("field").toString();

			final Object excludeBool = p.get("exclude");
			exclude = excludeBool != null ? (boolean)cosineBool : true;

			//Get the query vector embedding
			vector = p.get("vector");

			//Determine if raw comma-delimited vector or embedding was passed
			if(vector != null) {
			    final ArrayList<Object> tmp = (ArrayList<Object>) vector;
			    inputVector = new double[tmp.size()];
			    for (int i = 0; i < inputVector.length; i++) {
				Object obj = tmp.get(i);
				if (obj instanceof Double) {
				    inputVector[i] = ((Double)obj).doubleValue();
				} else if (obj instanceof Integer) {
				    inputVector[i] = ((Integer)obj).doubleValue();
				} else {
				    throw new IllegalArgumentException("Cannot convert vector to double array");
				}
			    }
			} else {
			    final Object encodedVector = p.get("encoded_vector");
			    if(encodedVector == null) {
				throw new IllegalArgumentException("Must have 'vector' or 'encoded_vector' as a parameter");
			    }
			    inputVector = Util.convertBase64ToArray((String) encodedVector);
			}

			// If cosine calculate the query vec norm
			if (cosine) {
			    queryVectorNorm = 0d;
			    // compute query inputVector norm once
			    for (double v : inputVector) {
				queryVectorNorm += v * v;
			    }
			    queryVectorNorm = Math.sqrt(queryVectorNorm);
			}
		    }

		    @Override
		    public ScoreScript newInstance(LeafReaderContext context) throws IOException {
			BinaryDocValues accessor = context.reader().getBinaryDocValues(field);
			if (accessor == null) {
			    /*
			     * the field and/or term don't exist in this segment,
			     * so always return 0
			     */
			    return new ScoreScript(p, lookup, context) {
				@Override
				public double execute() {
				    return 0.0d;
				}
			    };
			}
			return new ScoreScript(p, lookup, context) {
			    int currentDocID = -1;
			    // Use Lucene LeafReadContext to access binary values directly.
			    
			    @Override
			    public void setDocument(int docID) {
				/*
				 * advance has undefined behavior calling with
				 * a docid <= its current docid
				 */
				if (accessor.docID() < docID) {
				    try {
					accessor.advance(docID);
				    } catch (IOException e) {
					throw new UncheckedIOException(e);
				    }
				}
				currentDocID = docID;
			    }

			    @Override
			    public double execute() {
				if (currentDocID != accessor.docID()) {
				    /*
				     * advance moved past the current doc, so this doc
				     * has no occurrences of the term
				     */
				    return 0.0d;
				}
				
				
				final int inputVectorSize = inputVector.length;
				final byte[] bytes;

				try {
				    bytes = accessor.binaryValue().bytes;
				} catch (IOException e) {
				    return 0d;
				}


				final ByteArrayDataInput docVectorByteArray = new ByteArrayDataInput(bytes);

				docVectorByteArray.readVInt();

				final int docVectorLength = docVectorByteArray.readVInt(); // returns the number of bytes to read

				if (docVectorLength != inputVectorSize * 8) {
				    return 0d;
				}

				final int position = docVectorByteArray.getPosition();

				final DoubleBuffer doubleBuffer = ByteBuffer.wrap(bytes, position, docVectorLength).asDoubleBuffer();

				final double[] docVector = new double[inputVectorSize];
				doubleBuffer.get(docVector);

				double docVectorNorm = 0d;
				double score = 0d;

				// calculate dot product of document vector and query vector
				for (int i = 0; i < inputVectorSize; i++) {
          
				    score += docVector[i] * inputVector[i];

				    if (cosine) {
					docVectorNorm += docVector[i] * docVector[i];
				    }
				}

				// If cosine, calcluate cosine score
				if (cosine) {
					
				    if (docVectorNorm == 0 || queryVectorNorm == 0) return 0d;

				    score =  score / (Math.sqrt(docVectorNorm) * Math.sqrt(queryVectorNorm));
				}

				return score;
			    }
			};
		    }

		    @Override
		    public boolean needs_score() {
			return false;
		    }
		};
		return context.factoryClazz.cast(factory);
	    }
	    throw new IllegalArgumentException("Unknown script name " + scriptSource);
	}

	@Override
	public void close() {
	    // optionally close resources
	}
    }
}

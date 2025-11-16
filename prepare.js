import 'dotenv/config';

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
const filePath='./om-internal-docs.pdf'


export const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: "text-embedding-3-small",
});

const pinecone = new PineconeClient({
  apiKey: process.env.PINECONE_API_KEY,
});

const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX);

export const vectorStore = await PineconeStore.fromExistingIndex(
  embeddings,
  {
    pineconeIndex,
    maxConcurrency: 5,
  }
);

export async function indexTheDocument(filePath) {
  const loader = new PDFLoader(filePath, { splitPages: false });
  const doc = await loader.load();

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });

  const chunks = await textSplitter.splitText(doc[0].pageContent);

  const documents = chunks.map((chunk) => ({
    pageContent: chunk,
    metadata: doc[0].metadata,
  }));

  await vectorStore.addDocuments(documents);
}

indexTheDocument(filePath)
  .then(() => console.log("Indexing complete!"))
  .catch((err) => console.error("Indexing failed:", err));

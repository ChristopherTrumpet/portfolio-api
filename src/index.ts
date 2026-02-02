import { Hono } from "hono";
import { cors } from "hono/cors";
import { createClient, Client } from "@libsql/client/web";
import { bearerAuth } from "hono/bearer-auth";

import { streamText } from "hono/streaming";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

import context from "./data/context.json";
import { cosineSimilarity, createChunks, EmbeddedDoc } from "./utils/rag.js";
import { handle } from "hono/vercel";

type Bindings = {
  BOOKMARKS_DB_REMOTE_URL: string;
  BOOKMARKS_DB_APP_TOKEN: string;
  API_AUTH_TOKEN: string;
  GOOGLE_API_KEY: string;
};

type Variables = {
  db: Client;
};

const app = new Hono<{ Bindings: Bindings; Variables: Variables }>();

let cachedKnowledgeBase: EmbeddedDoc[] | null = null;

async function getKnowledgeBase(apiKey: string) {
  if (cachedKnowledgeBase) {
    return cachedKnowledgeBase;
  }

  console.log("Cold Start: Building Knowledge Base...");

  // 1. Generate text chunks from JSON
  const docs = createChunks(context);

  // 2. Initialize Embedding Model
  const embeddingsModel = new GoogleGenerativeAIEmbeddings({
    modelName: "text-embedding-004", // Low cost, high performance
    apiKey: apiKey,
  });

  // 3. Embed all chunks in parallel (Batch Request)
  const vectors = await embeddingsModel.embedDocuments(
    docs.map((d) => d.pageContent),
  );

  // 4. Store in memory
  cachedKnowledgeBase = docs.map((doc, i) => ({
    content: doc.pageContent,
    metadata: doc.metadata,
    embedding: vectors[i],
  }));

  console.log(`Knowledge Base Built: ${cachedKnowledgeBase.length} chunks.`);
  return cachedKnowledgeBase;
}

const initTurso = (url: string, token: string) => {
  return createClient({ url, authToken: token });
};

app.use(
  "*",
  cors({
    origin: "*",
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowHeaders: ["Content-Type", "Authorization"],
  }),
  async (c, next) => {
    if (c.req.path.includes("/chat")) {
      await next();
      return;
    }

    const url =
      c.env?.BOOKMARKS_DB_REMOTE_URL ?? process.env.BOOKMARKS_DB_REMOTE_URL;
    const token =
      c.env?.BOOKMARKS_DB_APP_TOKEN ?? process.env.BOOKMARKS_DB_APP_TOKEN;

    if (!url) {
      console.error("BOOKMARKS_DB_REMOTE_URL is not defined");
    } else {
      const db = initTurso(url, token);
      c.set("db", db);
    }

    await next();
  },
);

app.get("/bookmarks", async (c) => {
  const db = c.get("db");
  const { rows } = await db.execute("SELECT * FROM Bookmark");

  return c.json(rows);
});

app.post(
  "/bookmarks",
  async (c, next) => {
    const token = c.env?.API_AUTH_TOKEN ?? process.env.API_AUTH_TOKEN;
    const auth = bearerAuth({ token });
    return auth(c, next);
  },
  async (c) => {
    const db = c.get("db");

    try {
      const body = await c.req.json();
      const { title, author, url, comments, type } = body;

      if (!title || !url || !author) {
        return c.json({ error: "Title, Author, and URL are required" }, 400);
      }

      await db.execute({
        sql: "INSERT INTO Bookmark (title, author, url, comments, dateAdded, type) VALUES (?, ?, ?, ?, ?, ?)",
        args: [
          title,
          author,
          url,
          comments ?? "",
          new Date().toISOString(),
          type,
        ],
      });

      return c.json({ message: "Bookmark created successfully" }, 201);
    } catch (e: any) {
      return c.json(
        { error: "Failed to create bookmark", details: e.message },
        500,
      );
    }
  },
);

app.post("/chat", async (c) => {
  try {
    const body = await c.req.json();
    const { messages } = body;
    const lastMessage = messages[messages.length - 1].content;
    const apiKey = c.env?.GOOGLE_API_KEY ?? process.env.GOOGLE_API_KEY;

    if (!apiKey) {
      return c.json({ error: "Missing Google API Key" }, 500);
    }

    const knowledgeBase = await getKnowledgeBase(apiKey);

    // Embed the user's question
    const embeddingsModel = new GoogleGenerativeAIEmbeddings({
      modelName: "text-embedding-004",
      apiKey: apiKey,
    });
    const queryVector = await embeddingsModel.embedQuery(lastMessage);

    // Calculate similarity and sort
    const results = knowledgeBase
      .map((doc) => ({
        ...doc,
        score: cosineSimilarity(queryVector, doc.embedding),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5); // Take top 5 most relevant chunks

    // Combine chunks into a string
    const retrievedContext = results.map((r) => r.content).join("\n\n---\n\n");
    console.log(
      `🔍 Retrieved top ${results.length} chunks for query: "${lastMessage}"`,
    );

    const llm = new ChatGoogleGenerativeAI({
      model: "gemini-2.5-flash",
      apiKey: apiKey,
      temperature: 0.7,
    });

    const systemInstruction = new SystemMessage(
      `You are an interactive portfolio assistant.
       
       CONTEXT:
       ${retrievedContext}
       
       INSTRUCTIONS:
       - Use ONLY the provided context information. If the answer isn't there, say you don't know.
       - Link Formatting: [Title](URL) or [Email](mailto:...). No raw URLs.
       - Keep it concise and professional.
       ${context.system_instructions || ""}`,
    );

    const conversation = [
      systemInstruction,
      ...messages.map((msg: any) => new HumanMessage(msg.content)),
    ];

    return streamText(c, async (stream) => {
      const responseStream = await llm.stream(conversation);
      for await (const chunk of responseStream) {
        if (chunk.content && typeof chunk.content === "string") {
          await stream.write(chunk.content);
        }
      }
    });
  } catch (error) {
    console.error("Chat Error:", error);
    return c.json({ error: "Failed to generate response" }, 500);
  }
});

export const config = {
  runtime: "edge",
};

export default handle(app);

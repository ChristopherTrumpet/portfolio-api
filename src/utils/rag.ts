import { Document } from "@langchain/core/documents";

export function cosineSimilarity(vecA: number[], vecB: number[]): number {
  const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
  const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
  return dotProduct / (magnitudeA * magnitudeB);
}

export interface EmbeddedDoc {
  content: string;
  metadata: Record<string, any>;
  embedding: number[];
}

export function createChunks(data: any): Document[] {
  const docs: Document[] = [];

  // 1. PROFILE & CONTACT
  if (data.profile || data.contact) {
    docs.push(
      new Document({
        pageContent: `PROFILE & CONTACT:
      Name: ${data.profile?.name}
      Role: ${data.profile?.role}
      Bio: ${data.profile?.bio}
      Email: ${data.contact?.email}
      Phone: ${data.contact?.phone}
      Links: LinkedIn (${data.contact?.linkedin}), GitHub (${data.contact?.github})`,
        metadata: { type: "bio" },
      }),
    );
  }

  // 2. ACADEMICS: Courses
  // Each course gets its own chunk so queries like "Did he take Java?" find specific classes.
  if (data.academics?.courses) {
    data.academics.courses.forEach((c: any) => {
      docs.push(
        new Document({
          pageContent: `COURSE: ${c.identifier} - ${c.title} (${c.year})
        Description: ${c.description}
        Accomplishments: ${c.accomplishments?.join(", ")}
        Tech Stack: ${c.technology?.join(", ")}`,
          metadata: { type: "course", id: c.identifier },
        }),
      );
    });
  }

  // 3. ACADEMICS: Extracurriculars
  if (data.academics?.extracurriculars) {
    data.academics.extracurriculars.forEach((e: any) => {
      docs.push(
        new Document({
          pageContent: `EXTRACURRICULAR: ${e.title} at ${e.org}
        Description: ${e.description}
        Tech Stack: ${e.technology?.join(", ")}`,
          metadata: { type: "extracurricular" },
        }),
      );
    });
  }

  // 4. WORK EXPERIENCE
  // Your structure nests work under specific organizations (e.g. "envision_center")
  // We iterate through the keys in "work" to handle multiple jobs dynamically.
  if (data.work) {
    Object.keys(data.work).forEach((workKey) => {
      const job = data.work[workKey];

      // Projects within a job
      if (job.projects) {
        job.projects.forEach((p: any) => {
          docs.push(
            new Document({
              pageContent: `WORK PROJECT (${workKey}): ${p.name}
            Team Size: ${p.team_size}
            Description: ${p.description}
            Tech Stack: ${p.technology?.join(", ")}`,
              metadata: { type: "work_project", workplace: workKey },
            }),
          );
        });
      }

      // Events within a job
      if (job.events) {
        job.events.forEach((e: any) => {
          docs.push(
            new Document({
              pageContent: `WORK EVENT (${workKey}): ${e.name}
            Description: ${e.description}`,
              metadata: { type: "work_event", workplace: workKey },
            }),
          );
        });
      }
    });
  }

  // 5. PERSONAL PROJECTS
  if (data.projects) {
    data.projects.forEach((p: any) => {
      docs.push(
        new Document({
          pageContent: `PERSONAL PROJECT: ${p.title}
        Description: ${p.description}
        Tech Stack: ${p.technology?.join(", ")}
        Link: ${p.link}
        Category: ${p.category?.join(", ")}`,
          metadata: { type: "personal_project", title: p.title },
        }),
      );
    });
  }

  return docs;
}

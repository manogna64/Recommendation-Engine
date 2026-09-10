"""
AI Internship Recommendation Engine
Free Online AI & Data Science Internship — Task AI-SS-002
Student Support & Internship Management Recommendation

Implements:
  - Student Profile Management
  - Internship Database
  - Collaborative Filtering (user-based)
  - Content-Based Filtering (TF-IDF + cosine similarity)
  - Hybrid Recommendation (weighted combination)
  - Recommendation Evaluation (RMSE, Precision@K, Recall@K)
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


class InternshipRecommender:
    def __init__(self):
        self.students_df = self._load_students()
        self.internships_df = self._load_internships()
        self.ratings_df = self._load_ratings()
        self.user_item_matrix = self._build_user_item_matrix()

    # ------------------------------------------------------------------
    # Student Profile Management
    # ------------------------------------------------------------------
    def _load_students(self):
        data = {
            "student_id": ["S001", "S002", "S003", "S004", "S005", "S006"],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Farah"],
            "skills": [
                ["python", "machine learning", "data science"],
                ["java", "spring boot", "sql"],
                ["python", "django", "javascript"],
                ["python", "deep learning", "computer vision"],
                ["java", "android", "kotlin"],
                ["python", "nlp", "machine learning"],
            ],
            "domain": ["AI", "Web", "Web", "AI", "Mobile", "AI"],
            "experience_level": [
                "intermediate", "beginner", "intermediate",
                "advanced", "beginner", "intermediate",
            ],
        }
        return pd.DataFrame(data)

    def add_student(self, student_id, name, skills, domain, experience_level):
        """Add a new student profile to the system."""
        new_row = pd.DataFrame([{
            "student_id": student_id, "name": name, "skills": skills,
            "domain": domain, "experience_level": experience_level,
        }])
        self.students_df = pd.concat([self.students_df, new_row], ignore_index=True)
        return self.get_student_profile(student_id)

    def get_student_profile(self, student_id):
        match = self.students_df[self.students_df["student_id"] == student_id]
        return None if match.empty else match.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Internship Database
    # ------------------------------------------------------------------
    def _load_internships(self):
        data = {
            "internship_id": ["I001", "I002", "I003", "I004", "I005", "I006"],
            "title": [
                "AI Intern", "Java Developer", "Full Stack Developer",
                "Data Scientist", "Android Developer", "NLP Research Intern",
            ],
            "required_skills": [
                ["python", "machine learning", "tensorflow"],
                ["java", "spring boot", "sql"],
                ["python", "django", "javascript", "react"],
                ["python", "data science", "pandas", "scikit-learn"],
                ["java", "android", "kotlin"],
                ["python", "nlp", "machine learning", "spacy"],
            ],
            "domain": ["AI", "Web", "Web", "AI", "Mobile", "AI"],
            "duration": ["3 months", "6 months", "3 months", "6 months", "3 months", "3 months"],
        }
        return pd.DataFrame(data)

    def add_internship(self, internship_id, title, required_skills, domain, duration):
        """Add a new internship listing to the system."""
        new_row = pd.DataFrame([{
            "internship_id": internship_id, "title": title,
            "required_skills": required_skills, "domain": domain, "duration": duration,
        }])
        self.internships_df = pd.concat([self.internships_df, new_row], ignore_index=True)
        return self.get_internship(internship_id)

    def get_internship(self, internship_id):
        match = self.internships_df[self.internships_df["internship_id"] == internship_id]
        return None if match.empty else match.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Ratings / interaction data
    # ------------------------------------------------------------------
    def _load_ratings(self):
        data = {
            "student_id": [
                "S001", "S001", "S002", "S002", "S003",
                "S003", "S004", "S004", "S005", "S006", "S006",
            ],
            "internship_id": [
                "I001", "I003", "I002", "I004", "I001",
                "I003", "I001", "I004", "I002", "I001", "I006",
            ],
            "rating": [5, 4, 5, 3, 4, 5, 5, 4, 3, 4, 5],
        }
        return pd.DataFrame(data)

    def _build_user_item_matrix(self):
        return self.ratings_df.pivot(
            index="student_id", columns="internship_id", values="rating"
        ).fillna(0)

    # ------------------------------------------------------------------
    # Collaborative Filtering (user-based)
    # ------------------------------------------------------------------
    def _predict_rating(self, student_id, internship_id, matrix=None):
        """Predict a student's rating for an internship as a similarity-weighted
        average of ratings given by the most similar students."""
        matrix = self.user_item_matrix if matrix is None else matrix
        if student_id not in matrix.index or internship_id not in matrix.columns:
            return 0.0

        target = matrix.loc[student_id].values.reshape(1, -1)
        sims = cosine_similarity(target, matrix.values)[0]

        num, den = 0.0, 0.0
        for idx, other_id in enumerate(matrix.index):
            if other_id == student_id:
                continue
            rating = matrix.loc[other_id, internship_id]
            if rating > 0 and sims[idx] > 0:
                num += sims[idx] * rating
                den += sims[idx]

        return round(num / den, 3) if den > 0 else 0.0

    def collaborative_filtering(self, student_id, top_n=3):
        """Recommend internships using user-based collaborative filtering."""
        if student_id not in self.user_item_matrix.index:
            return "Student not found"

        student_ratings = self.user_item_matrix.loc[student_id]
        unrated = [c for c in self.user_item_matrix.columns if student_ratings[c] == 0]

        scored = [(iid, self._predict_rating(student_id, iid)) for iid in unrated]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)

        result = []
        for internship_id, score in scored[:top_n]:
            internship = self.internships_df[
                self.internships_df["internship_id"] == internship_id
            ].iloc[0]
            result.append({
                "internship_id": internship_id,
                "title": internship["title"],
                "domain": internship["domain"],
                "predicted_score": score,
            })
        return result

    # ------------------------------------------------------------------
    # Content-Based Filtering
    # ------------------------------------------------------------------
    def content_based_filtering(self, student_id, top_n=3):
        """Recommend internships using TF-IDF similarity between a student's
        skill/domain profile and each internship's requirements."""
        if student_id not in self.students_df["student_id"].values:
            return "Student not found"

        student = self.students_df[self.students_df["student_id"] == student_id].iloc[0]

        internship_texts = [
            " ".join(row["required_skills"]) + " " + row["domain"]
            for _, row in self.internships_df.iterrows()
        ]

        vectorizer = TfidfVectorizer()
        internship_vectors = vectorizer.fit_transform(internship_texts)

        student_text = " ".join(student["skills"]) + " " + student["domain"]
        student_vector = vectorizer.transform([student_text])

        sims = cosine_similarity(student_vector, internship_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:top_n]

        result = []
        for idx in top_indices:
            internship = self.internships_df.iloc[idx]
            result.append({
                "internship_id": internship["internship_id"],
                "title": internship["title"],
                "domain": internship["domain"],
                "similarity_score": round(float(sims[idx]), 3),
            })
        return result

    # ------------------------------------------------------------------
    # Hybrid Recommendation
    # ------------------------------------------------------------------
    def hybrid_recommendation(self, student_id, top_n=3, weight_collab=0.6, weight_content=0.4):
        """Combine collaborative and content-based scores into one ranked list."""
        collab_recs = self.collaborative_filtering(student_id, top_n * 2)
        content_recs = self.content_based_filtering(student_id, top_n * 2)
        if isinstance(collab_recs, str) or isinstance(content_recs, str):
            return "Student not found"

        combined = {}
        for rec in collab_recs:
            combined[rec["internship_id"]] = {
                "title": rec["title"], "domain": rec["domain"],
                "score": rec["predicted_score"] * weight_collab,
            }
        for rec in content_recs:
            if rec["internship_id"] in combined:
                combined[rec["internship_id"]]["score"] += rec["similarity_score"] * weight_content
            else:
                combined[rec["internship_id"]] = {
                    "title": rec["title"], "domain": rec["domain"],
                    "score": rec["similarity_score"] * weight_content,
                }

        sorted_recs = sorted(combined.items(), key=lambda x: x[1]["score"], reverse=True)[:top_n]
        return [
            {
                "internship_id": iid, "title": d["title"],
                "domain": d["domain"], "combined_score": round(d["score"], 3),
            }
            for iid, d in sorted_recs
        ]

    # ------------------------------------------------------------------
    # Recommendation Evaluation (RMSE, Precision@K, Recall@K)
    # ------------------------------------------------------------------
    def evaluate(self, top_n=3, relevance_threshold=4):
        """Leave-one-out evaluation over every known rating.

        For each (student, internship, rating) triple:
          1. Temporarily hide that rating from the user-item matrix.
          2. Predict it with collaborative filtering -> used for RMSE.
          3. Generate top-N recommendations from the remaining data and check
             whether the held-out internship appears -> used for Precision/Recall.
        """
        squared_errors = []
        hits, total_relevant, total_recommended = 0, 0, 0

        for _, row in self.ratings_df.iterrows():
            sid, iid, true_rating = row["student_id"], row["internship_id"], row["rating"]

            leave_one_out_matrix = self.user_item_matrix.copy()
            leave_one_out_matrix.loc[sid, iid] = 0

            predicted = self._predict_rating(sid, iid, matrix=leave_one_out_matrix)
            if predicted > 0:
                squared_errors.append((predicted - true_rating) ** 2)

            is_relevant = true_rating >= relevance_threshold
            if is_relevant:
                total_relevant += 1

            unrated = [c for c in leave_one_out_matrix.columns if leave_one_out_matrix.loc[sid, c] == 0]
            scored = [(c, self._predict_rating(sid, c, matrix=leave_one_out_matrix)) for c in unrated]
            scored = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
            recommended_ids = [c for c, _ in scored]
            total_recommended += len(recommended_ids)

            if is_relevant and iid in recommended_ids:
                hits += 1

        rmse = round(float(np.sqrt(np.mean(squared_errors))), 3) if squared_errors else None
        precision = round(hits / total_recommended, 3) if total_recommended else 0.0
        recall = round(hits / total_relevant, 3) if total_relevant else 0.0

        return {
            "RMSE": rmse,
            "Precision@{}".format(top_n): precision,
            "Recall@{}".format(top_n): recall,
            "evaluated_ratings": len(self.ratings_df),
        }


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
def main():
    recommender = InternshipRecommender()

    student_id = "S001"
    profile = recommender.get_student_profile(student_id)
    print(f"Student Profile — {student_id}: {profile['name']} | "
          f"skills={profile['skills']} | domain={profile['domain']} | "
          f"level={profile['experience_level']}\n")

    print(f"Recommendations for {student_id}:\n")

    print("Collaborative Filtering:")
    for r in recommender.collaborative_filtering(student_id):
        print(f"  {r['title']:<22} ({r['domain']})  predicted_score={r['predicted_score']}")

    print("\nContent-Based Filtering:")
    for r in recommender.content_based_filtering(student_id):
        print(f"  {r['title']:<22} ({r['domain']})  similarity={r['similarity_score']}")

    print("\nHybrid Recommendation:")
    for r in recommender.hybrid_recommendation(student_id):
        print(f"  {r['title']:<22} ({r['domain']})  combined_score={r['combined_score']}")

    print("\nEvaluation (leave-one-out over all known ratings):")
    metrics = recommender.evaluate(top_n=3)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nAdding a new student profile:")
    new_student = recommender.add_student(
        "S007", "Grace", ["python", "computer vision", "opencv"], "AI", "intermediate"
    )
    print(f"  Added: {new_student}")
    print("  New recommendations for S007:")
    for r in recommender.content_based_filtering("S007"):
        print(f"    {r['title']:<22} ({r['domain']})  similarity={r['similarity_score']}")


if __name__ == "__main__":
    main()

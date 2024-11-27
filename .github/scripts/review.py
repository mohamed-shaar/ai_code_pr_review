#!/usr/bin/env python3
import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import requests
from github import Github
from llama_cpp import Llama

class CodeReviewer:
    def __init__(self):
        """Initialize the code reviewer with GitHub and LLM configurations."""
        # GitHub Authentication
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = os.getenv('GITHUB_REPOSITORY')
        self.pr_number = int(os.getenv('PR_NUMBER', 0))

        # Review Configuration
        self.max_files = int(os.getenv('MAX_FILES_TO_REVIEW', 10))
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', 1000))

        # Initialize GitHub and model
        self.gh_client = Github(self.github_token)
        self.repo = self.gh_client.get_repo(self.repo_name)
        self.pr = self.repo.get_pull(self.pr_number)

        # Model path
        self.model_path = str(Path.home() / '.cache/models/codellama-7b.Q4_K_M.gguf')

        # Validate model exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

    def _load_model(self):
        """Load the LLM with optimized settings."""
        try:
            return Llama(
                model_path=self.model_path,
                n_ctx=2048,  # Context window
                n_threads=os.cpu_count(),  # Use all CPU threads
                n_batch=512  # Batch size for processing
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def get_file_content(self, file_path: str) -> str:
        """
        Retrieve file content from the repository.

        Args:
            file_path (str): Path to the file in the repository

        Returns:
            str: File content or empty string if unable to read
        """
        try:
            content_file = self.repo.get_contents(file_path, ref=self.pr.head.sha)
            return content_file.decoded_content.decode('utf-8')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""

    def analyze_code(self, code: str, file_path: str) -> str:
        """
        Analyze code using the LLM.

        Args:
            code (str): Code content to analyze
            file_path (str): Path of the file being analyzed

        Returns:
            str: AI-generated code review
        """
        model = self._load_model()

#Provide a detailed code review for the following {Path(file_path).suffix} file:
#File: {file_path}
        prompt = f"""
        

        Provide a detailed code review for this Flutter code. 

        Focus on:
            1. Code quality and best practices specific to Flutter and Dart
            2. Potential performance bottlenecks or widget rendering issues
            3. Security considerations in mobile app development
            4. Recommended architectural or implementation improvements

        Please include specific, actionable feedback with code suggestions where applicable.
        Code:
        ```
        {code}
        ```

        Focus on:
        1. Code quality and best practices
        2. Potential bugs or performance issues
        3. Security considerations
        4. Recommended improvements

        Provide concise, actionable feedback.
        """
        print(prompt)

        try:
            response = model.create_completion(
                prompt,
                max_tokens=1024,
                temperature=0.3,
                stop=["```"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error analyzing code: {e}")
            return f"Unable to generate review due to error: {e}"

    def create_pr_comment(self, reviews: List[Dict[str, Any]]):
        """
        Create a comprehensive PR comment with review findings.

        Args:
            reviews (List[Dict]): List of file reviews
        """
        comment_body = "## 🤖 AI Code Review\n\n"

        for review in reviews:
            comment_body += f"### 📄 File: `{review['file_path']}`\n\n"
            comment_body += f"{review['review']}\n\n"

        comment_body += "\n---\n"
        comment_body += "_Automated review by CodeLlama. Verify suggestions carefully._"

        # Post or update comment
        existing_comments = list(self.pr.get_issue_comments())
        for comment in existing_comments:
            if "🤖 AI Code Review" in comment.body:
                comment.edit(comment_body)
                return

        self.pr.create_issue_comment(comment_body)

    def review_pull_request(self):
        """
        Main method to review pull request files.
        Processes files, generates reviews, and posts comments.
        """
        reviews = []

        # Get changed files
        changed_files = self.pr.get_files()

        for file in changed_files:
            # Filter for specific file types
            if not file.filename.endswith(('.dart', '.yaml', '.json')):
                continue

            # Skip large files
            if file.additions > self.max_file_size:
                continue

            # Retrieve and analyze file content
            content = self.get_file_content(file.filename)
            if not content:
                continue

            review = self.analyze_code(content, file.filename)
            reviews.append({
                'file_path': file.filename,
                'review': review
            })

            # Stop if max files reviewed
            if len(reviews) >= self.max_files:
                break

        # Post reviews
        if reviews:
            self.create_pr_comment(reviews)
        else:
            print("No files to review.")

def main():
    """Entry point for the code review script."""
    try:
        reviewer = CodeReviewer()
        reviewer.review_pull_request()
    except Exception as e:
        print(f"Critical error in code review: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
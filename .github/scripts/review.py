class CodeReviewer:
    def __init__(self):
        # ... previous init code ...

        # Add performance optimizations
        self.max_files = int(os.getenv('MAX_FILES_TO_REVIEW', 10))
        self.max_file_size = int(os.getenv('MAX_FILE_SIZE', 1000))

        # Pre-load model once
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize model with optimized settings."""
        model_path = str(Path.home() / '.cache/models/codellama-7b.Q4_K_M.gguf')
        return Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=os.cpu_count(),  # Use all available CPU threads
            n_batch=512,  # Increase batch size
            vocab_only=False,
            embedding=False  # Disable if not needed
        )

    def review_pr(self):
        """Main function to review the PR with optimizations."""
        reviews = []
        files_reviewed = 0
        start_time = time.time()

        # Get changed files and sort by size
        changed_files = sorted(
            self.pr.get_files(),
            key=lambda x: x.changes
        )[:self.max_files]

        for file in changed_files:
            if files_reviewed >= self.max_files:
                break

            # Skip large files
            if file.changes > self.max_file_size:
                continue

            # Process file and add timing metrics
            file_start = time.time()
            review = self.analyze_flutter_code(content, file.filename)
            file_time = time.time() - file_start

            reviews.append({
                'filename': file.filename,
                'review': review,
                'time_taken': file_time
            })

            files_reviewed += 1

        # Add timing summary to review
        total_time = time.time() - start_time
        timing_summary = f"\n### ⏱️ Performance Metrics\n"
        timing_summary += f"- Total review time: {total_time:.2f}s\n"
        timing_summary += f"- Files reviewed: {files_reviewed}\n"

        self.create_review_comment(reviews, timing_summary)
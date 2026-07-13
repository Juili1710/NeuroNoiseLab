import pandas as pd
import matplotlib.pyplot as plt


class JuryAnalysis:

    def __init__(self):

        self.df = None

    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------

    def load_csv(self, file_path):

        self.df = pd.read_csv(file_path)

        print(
            f"Loaded {len(self.df)} responses"
        )

        return self.df

    # --------------------------------------------------
    # Gender Distribution
    # --------------------------------------------------

    def gender_pie(self):

        if self.df is None:
            return

        counts = self.df["gender"].value_counts()

        labels = [
            f"{gender}\n(n={count})"
            for gender, count in counts.items()
        ]

        plt.figure(figsize=(7,7))

        plt.pie(
            counts,
            labels=labels,
            autopct="%1.1f%%"
        )

        plt.title(
            f"Gender Distribution\nParticipants = {self.df['listener_id'].nunique()}"
        )
        self.save_plot("gender_distribution.png")
        plt.show()
        


    # --------------------------------------------------
    # Age Distribution
    # --------------------------------------------------

    def age_histogram(self):

        if self.df is None:
            return
        # Demographic statistics
        n_participants = self.df["listener_id"].nunique()

        median_age = self.df["age"].median()
        mean_age = self.df["age"].mean()
        std_age = self.df["age"].std()

        
        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["age"],
            bins=5
        )
        plt.title(
            f"Age Distribution\n"
            f"Participants = {n_participants} | "
            f"Median Age = {median_age:.0f} yrs"
            f"Age = {mean_age:.1f} ± {std_age:.1f} yrs "
        )
        plt.xlabel("Age")

        plt.ylabel("Participants")

        

        plt.grid(True)
        self.save_plot("age_distribution.png")
        plt.show()

    # --------------------------------------------------
    # Annoyance Distribution
    # --------------------------------------------------

    def annoyance_histogram(self):

        if self.df is None:
            return

        plt.figure(figsize=(8,5))

        self.df["Q4"].value_counts()\
            .sort_index()\
            .plot(
                kind="bar"
            )

        plt.xlabel(
            "Annoyance Rating"
        )

        plt.ylabel(
            "Number of Responses"
        )

        plt.title(
            "Annoyance Rating Distribution"
        )

        plt.grid(True)

        self.save_plot("annoyance_distribution.png")
        plt.show()

    # --------------------------------------------------
    # Pleasantness Distribution
    # --------------------------------------------------

    def pleasantness_histogram(self):

        if self.df is None:
            return

        plt.figure(figsize=(8,5))

        self.df["Q5"].value_counts()\
            .sort_index()\
            .plot(
                kind="bar"
            )

        plt.xlabel(
            "Pleasantness Rating"
        )

        plt.ylabel(
            "Number of Responses"
        )

        plt.title(
            "Pleasantness Rating Distribution"
        )

        plt.grid(True)

        self.save_plot("pleasantness_distribution.png")
        plt.show()

    # --------------------------------------------------
    # Mean Ratings Per Question
    # --------------------------------------------------

    def mean_ratings(self):

        if self.df is None:
            return

        ratings = [
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Q6",
            "Q7"
        ]

        means = self.df[
            ratings
        ].mean()

        plt.figure(figsize=(10,5))

        means.plot(
            kind="bar"
        )

        plt.ylabel(
            "Average Rating"
        )

        plt.title(
            "Mean Jury Ratings"
        )

        plt.grid(True)

        self.save_plot("mean_ratings.png")
        plt.show()

    # --------------------------------------------------
    # Mean Annoyance Per Stimulus
    # --------------------------------------------------

    def annoyance_per_stimulus(self):

        if self.df is None:
            return

        means = self.df.groupby(
            "stimulus"
        )["Q4"].mean()

        plt.figure(figsize=(12,5))

        means.sort_values().plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean Annoyance"
        )

        plt.title(
            "Average Annoyance Per Stimulus"
        )

        plt.tight_layout()

        self.save_plot("annoyance_per_stimulus.png")
        plt.show()

    # --------------------------------------------------
    # Mean Pleasantness Per Stimulus
    # --------------------------------------------------

    def pleasantness_per_stimulus(self):

        if self.df is None:
            return

        means = self.df.groupby(
            "stimulus"
        )["Q5"].mean()

        plt.figure(figsize=(12,5))

        means.sort_values().plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean Pleasantness"
        )

        plt.title(
            "Average Pleasantness Per Stimulus"
        )

        plt.tight_layout()
        self.save_plot("pleasantness_per_stimulus.png")
        plt.show()

    # --------------------------------------------------
    # Stimulus Comparison
    # --------------------------------------------------

    def stimulus_comparison(self):

        if self.df is None:
            return

        summary = self.df.groupby(
            "stimulus"
        )[["Q4","Q5"]].mean()

        plt.figure(figsize=(12,6))

        summary.plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean Rating"
        )

        plt.title(
            "Annoyance vs Pleasantness"
        )

        plt.tight_layout()

        self.save_plot("annoyance_vs_pleasantness.png")
        plt.show()
    def detectability_per_stimulus(self):


        if self.df is None:
            return

        means = self.df.groupby(
            "stimulus"
        )["Q1"].mean()

        plt.figure(figsize=(10,5))

        means.sort_values(
            ascending=False
        ).plot(
            kind="bar"
        )

        plt.ylabel(
            "Mean Detectability"
        )

        plt.xlabel(
            "Stimulus"
        )

        plt.title(
            "Average Detectability per Stimulus"
        )

        plt.tight_layout()

        self.save_plot("detectability_per_stimulus.png")
        plt.show()

    def save_plot(self, filename):

        from pathlib import Path

        results_dir = Path("Results")
        results_dir.mkdir(exist_ok=True)

        plt.savefig(
            results_dir / filename,
            dpi=300,
            bbox_inches="tight"
        )
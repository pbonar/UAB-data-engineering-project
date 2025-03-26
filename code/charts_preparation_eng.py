import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global configuration using correct relative paths
PLOT_CONFIG = {
    'output_dir': os.path.join(SCRIPT_DIR, "../../UAB-data-engineering-project/charts/"),
    'data_dir': os.path.join(SCRIPT_DIR, "../../analysis-of-psychoactive-substance-use/data/"),
    'dpi': 300,
    'bbox_inches': 'tight',
    'figsize': (10, 6),
    'fontsize': {
        'title': 14,
        'axis': 12,
        'legend': 10
    }
}

def configure_output(filename: str) -> str:
    """Configure and create output directory, return full output path."""
    os.makedirs(PLOT_CONFIG['output_dir'], exist_ok=True)
    return os.path.join(PLOT_CONFIG['output_dir'], filename)

def save_plot(filename: str) -> None:
    """Save plot with standardized configuration."""
    output_path = configure_output(filename)
    print(f"Attempting to save to: {output_path}")
    plt.savefig(
        output_path,
        dpi=PLOT_CONFIG['dpi'],
        bbox_inches=PLOT_CONFIG['bbox_inches']
    )
    print(f"✓ Successfully saved plot to: {output_path}")
    plt.close()

def load_data() -> pd.DataFrame:
    """Load data with proper path handling."""
    data_path = os.path.join(PLOT_CONFIG['data_dir'], "NSDUH_2022_selected_columns_validated.csv")
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    
    return pd.read_csv(data_path)

def histograph_age(df: pd.DataFrame) -> None:
    """Create a histogram visualization of age group distribution."""
    # Plot configuration
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon',
             'lightseagreen', 'lightpink', 'lightsteelblue']
    labels = ['4 - 18-20 years',
             '5 - 21-23 years',
             '6 - 24-25 years',
             '7 - 26-29 years',
             '8 - 30-34 years',
             '9 - 35-49 years',
             '10 - 50-64 years',
             '11 - 65+ years']

    age_counts = df['AGE3'].value_counts().sort_index()

    plt.figure(figsize=PLOT_CONFIG['figsize'])
    bars = plt.bar(age_counts.index, age_counts.values, color=colors[:len(age_counts)])

    plt.xticks(age_counts.index, range(4, 12), rotation=45, ha='right')
    plt.ylabel('Number of Participants', fontsize=PLOT_CONFIG['fontsize']['axis'])
    plt.xlabel('Age Group', fontsize=PLOT_CONFIG['fontsize']['axis'])
    plt.title('Distribution of Participants by Age Group (AGE3)', 
             fontsize=PLOT_CONFIG['fontsize']['title'])
    plt.legend(bars, labels, loc='upper left', fontsize=PLOT_CONFIG['fontsize']['legend'])

    save_plot("histogram_age3_eng.jpg")
    plt.tight_layout()
    plt.show()

def histograph_coutyp4(df: pd.DataFrame) -> None:
    """Create histogram of place types."""
    # Data preparation
    groups = {
        'Large Metro': df[df['COUTYP4'] == 1],
        'Small Metro': df[df['COUTYP4'] == 2],
        'Non-Metro': df[df['COUTYP4'] == 3]
    }
    counts = [len(g) for g in groups.values()]
    colors = ['skyblue', 'lightgreen', 'lightsalmon']
    labels = ['Large Metro', 'Small Metro', 'Non-Metro']

    # Create plot
    plt.figure(figsize=PLOT_CONFIG['figsize'])
    bars = plt.bar(range(3), counts, color=colors, width=0.5)
    
    # Add annotations
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval), 
                ha='center', va='bottom')

    # Configure plot
    plt.xticks(range(3), labels)
    plt.title('Distribution of Participants by Residence Type (COUTYP4)',
             fontsize=PLOT_CONFIG['fontsize']['title'])
    plt.ylabel('Number of Participants', fontsize=PLOT_CONFIG['fontsize']['axis'])

    save_plot("histogram_coutyp4_eng.jpg")
    plt.tight_layout()
    plt.show()

# Function for creating a histogram of sexual identity
def histograph_sexident(df: pd.DataFrame) -> None:
    # Dividing the rows into groups
    homo = df[df['SEXIDENT'] == 1]
    hetero = df[df['SEXIDENT'] == 2]
    bi = df[df['SEXIDENT'] == 3]

    # Creating the histogram
    plt.hist([homo['SEXIDENT'], hetero['SEXIDENT'], bi['SEXIDENT']], bins=range(1, 5), rwidth=0)

    # Adding ticks, titles and labels
    plt.xticks([1, 2, 3], ['Heterosexual', 'Homosexual', 'Bisexual'])
    plt.title('Distribution of Participants by Sexual Orientation (SEXIDENT)')
    plt.ylabel('Number of Participants')

    # Adjusting the bars
    counts = [homo.shape[0], hetero.shape[0], bi.shape[0]]
    bars = plt.bar(range(1, 4), counts, color=['skyblue', 'lightgreen', 'lightsalmon'], width=0.5, align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, int(yval), ha='center', va='bottom')

    # Saving and displaying the plot
    save_plot("histogram_sexident_eng.jpg")
    plt.tight_layout()
    plt.show()

# Function for creating a histogram of gender
def histograph_irsex(df: pd.DataFrame) -> None:
    # Dividing the rows into groups
    male = df[df['IRSEX'] == 1]
    female = df[df['IRSEX'] == 2]

    # Creating the histogram
    plt.hist([male['IRSEX'], female['IRSEX']], bins=range(1, 4), color=['skyblue', 'pink'], rwidth=0.3, align='left')

    # Adding ticks, titles and labels
    plt.xticks([1, 2], ['Male', 'Female'])
    plt.title('Distribution of participants by gender (IRSEX)')
    plt.ylabel('Number of participants')

    # Adjusting the bars
    bars = plt.bar(range(1, 3), [male.shape[0], female.shape[0]], color=['skyblue', 'pink'], width=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    # Saving and displaying the plot
    save_plot("histogram_irsex_eng.jpg")
    plt.tight_layout()
    plt.show()

# Function for creating a histogram of ethnic background
def histograph_newrace2(df: pd.DataFrame) -> None:
    # Defining colors and labels for ethnic groups
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'lightseagreen', 'lightpink']
    labels = [
        '1 - White',
        '2 - Black or African American',
        '3 - Native American',
        '4 - Native Hawaiian/Pacific Islander',
        '5 - Asian',
        '6 - More than one race',
        '7 - Hispanic/Latino'
    ]

    # Getting ethnic background and sorting them
    race_counts = df['NEWRACE2'].value_counts().sort_index()
    bars = plt.bar(race_counts.index, race_counts.values, color=colors[:len(race_counts)])

    # Setting ticks, labels, and title
    plt.xticks(race_counts.index, range(1, 8), rotation=45, ha='right')
    plt.ylabel('Number of participants')
    plt.title('Distribution of participants by ethnic origin (NEWRACE2)')
    plt.legend(bars, labels, loc='upper right')

    # Saving and displaying the plot
    save_plot("histogram_newrace2_eng.jpg")
    plt.tight_layout()
    plt.show()

# Function for creating a histogram of income levels
def histograph_income(df: pd.DataFrame) -> None:
    # Defining colors and labels for income levels
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightsalmon', 'lightseagreen', 'lightpink',
              'lightsteelblue', 'lightyellow', 'lightgrey', 'lightcyan']
    labels = ['1 - Less than $20,000',
          '2 - $20,000-$49,999',
          '3 - $50,000-$74,999', 
          '4 - More than $75,000']

    # Getting counts and sorting them
    income_counts = df['INCOME'].value_counts().sort_index()

    # Creating bars for the histogram
    bars = plt.bar(income_counts.index, income_counts.values, color=colors[:len(income_counts)])

    # Setting ticks, labels, and title
    plt.xticks(income_counts.index, range(1, 5), ha='right')
    plt.ylabel('Number of Participants')
    plt.xlabel('Total Annual Family Income') 
    plt.title('Distribution of Participants by Income Group (INCOME)')

    # Creating legend
    plt.legend(bars, labels, loc='upper left')

    # Saving and displaying the plot
    save_plot("histograph_income_eng.jpg")
    plt.tight_layout()
    plt.show()

# Function for creating a histogram of drug usage
def histograph_drug(df: pd.DataFrame, drug_column_name: str, drug_name: str, drug_values: List[int], drug_labels: List[str]) -> None:
    # Dividing the rows into groups
    taken = df[df[drug_column_name] == 1]
    not_taken = df[df[drug_column_name] == 2]

    # Creating the histogram
    plt.hist([taken[drug_column_name], not_taken[drug_column_name]], bins=range(1, 4), color=['lightsalmon', 'skyblue'],
             rwidth=0.3, align='left')

    # Adding ticks, titles and labels
    plt.xticks(drug_values, drug_labels)
    plt.title(f'Distribution of participants by {drug_name} usage ({drug_column_name})')
    plt.ylabel('Number of participants')

    # Adjusting the bars
    bars = plt.bar(range(1, 3), [taken.shape[0], not_taken.shape[0]], color=['lightsalmon', 'skyblue'], width=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.1, round(yval, 2), ha='center', va='bottom')

    # Saving and displaying the plot
    save_plot(f"histographs_drug_{drug_name}_eng.jpg")
    plt.tight_layout()
    plt.show()

def main() -> None:    
    try:
        df = load_data()

        # Generating histograms of demographic variables
        histograph_age(df)
        histograph_coutyp4(df)
        histograph_sexident(df)
        histograph_irsex(df)
        histograph_newrace2(df)
        histograph_income(df)

        # Generating histograms of substance use variables
        drug_config = [
            ('CIGFLAG', 'Cigarettes', [1, 2], ['Used', 'Never used']),
            ('ALCFLAG', 'Alcohol', [1, 2], ['Used', 'Never used']),
            ('MJEVER', 'Marijuana', [1, 2], ['Used', 'Never used']),
            ('COCEVER', 'Cocaine', [1, 2], ['Used', 'Never used']),
            ('HEREVER', 'Heroin', [1, 2], ['Used', 'Never used']),
            ('LSD', 'LSD', [1, 2], ['Used', 'Never used'])
        ]
        
        for col, name, vals, labels in drug_config:
            histograph_drug(df, col, name, vals, labels)

    except FileNotFoundError as e:
        print(f"✗ Error: {str(e)}")
        print("Please verify:")
        print(f"1. The file exists at {PLOT_CONFIG['data_dir']}")
        print(f"2. The script is running from the correct directory")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {SCRIPT_DIR}")
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- Page setup ---
st.set_page_config(page_title="NHL Defensemen: Data Analysis", layout="wide")
st.title("🏒 NHL Defensemen: Data Analysis")

# --- Load Data ---
@st.cache_data
def load_data():
    hockey = pd.read_excel('SS.xlsx')
    bio = pd.read_excel('Bio.xlsx')
    hockey_filtered = hockey.drop(columns=['Pos', 'Season', 'S%'], errors='ignore')
    bio_filtered = bio[['Ctry', 'Ht', 'Wt', 'Draft Yr', 'Round', 'Overall']]
    data = pd.concat([hockey_filtered, bio_filtered], axis=1)

    # Add Div and Conf columns
    ATL = ['BOS', 'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TBL', 'TOR']
    MET = ['CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WSH']
    CEN = ['CHI', 'COL', 'DAL', 'MIN', 'NSH', 'STL', 'UTA', 'WPG']
    PAC = ['ANA', 'CGY', 'EDM', 'LAK', 'SJS', 'SEA', 'VAN', 'VGK']

    def get_div(team):
        if team in ATL:
            return 'ATL'
        elif team in MET:
            return 'MET'
        elif team in CEN:
            return 'CEN'
        elif team in PAC:
            return 'PAC'
        else:
            return 'Unknown'

    def get_conf(team):
        if team in ATL or team in MET:
            return 'EC'
        elif team in CEN or team in PAC:
            return 'WC'
        else:
            return 'Unknown'

    if 'Team' in data.columns:
        data['Div'] = data['Team'].apply(get_div)
        data['Conf'] = data['Team'].apply(get_conf)
    else:
        data['Div'] = 'Unknown'
        data['Conf'] = 'Unknown'
    return data
df = load_data()

# --- Sidebar Navigation ---
st.sidebar.image("NHL-Logo.png", use_container_width=True)
st.sidebar.header("Navigation")

# Define options for two separate radio groups
main_options = [
    "About this App",
    "Dataset Overview",
    "Class Imbalance",
    "Missing Values",
    "Correlation Analysis",
    "Scatter Plots",
    "Wrapping Up",
]

project_options = [
    "Data Preperation + Collection",
    "EDA and Visulization",
    "Data Processing and Feature Engineering",
    "Model Development and Evaluation",
]

def _set_main_page():
    st.session_state['page'] = st.session_state.get('main_radio')

def _set_project_page():
    st.session_state['page'] = st.session_state.get('project_radio')

# initialize session state keys if missing
if 'page' not in st.session_state:
    st.session_state['page'] = main_options[0]
    st.session_state['main_radio'] = main_options[0]
    st.session_state['project_radio'] = project_options[0]

# First radio group (primary app sections)
main_page = st.sidebar.radio("Midterm Project Pages", main_options, key='main_radio', index=main_options.index(st.session_state.get('main_radio')) if st.session_state.get('main_radio') in main_options else 0, on_change=_set_main_page)

st.sidebar.markdown("---")

# Second radio group (final project pages)
project_page = st.sidebar.radio("Final Project Pages", project_options, key='project_radio', index=project_options.index(st.session_state.get('project_radio')) if st.session_state.get('project_radio') in project_options else 0, on_change=_set_project_page)

# The active page is stored in session_state['page'] (set by callbacks above)
page = st.session_state.get('page', main_options[0])

if page == "About this App":
    st.markdown("""
        <div style="background-color:#f0f2f6;padding:32px 24px 24px 24px;border-radius:12px;margin-bottom:24px;">
            <h2 style="color:#1a202c;margin-bottom:8px;">Get ready for Puck Drop!</h2>
            <p style="font-size:1.1rem;color:#444;line-height:1.6;margin-bottom:18px;">
                The main focus of this project is to use <b>Initial Data Analysis (IDA)</b> and <b>Exploratory Data Analysis (EDA)</b> to preprocess two raw datasets containing NHL defenseman data from the last year.<br><br>
                The goal is to get the data to a point where we can see who is producing the most points and what features correlate with point production. This will set the stage for my semester-end project: <b>predicting points using linear regression</b>.
            </p>
            <ul style="font-size:1.05rem;color:#333;margin-left:1.2em;">
                <li>You will find <b>interactive graphs</b> throughout the app to help you explore the data.</li>
                <li>There are <b>dropdown text boxes</b> on each page, providing explanations and insights for every analysis.</li>
            </ul>
            <div style="margin-top:24px;color:#888;font-size:0.98rem;">
                <i>Created by <b>Keshavi Dave</b> for her CMSE Midterm Project</i>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Dataset Overview ---
if page == "Dataset Overview":
    st.header("Dataset Overview")

    # Metric boxes with gray stripes
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.markdown('<div style="display:flex;align-items:center;background:#fff;border-radius:8px;box-shadow:0 1px 4px #e5e7eb;margin-bottom:8px;">'
                    '<div style="width:8px;height:48px;background:#e5e7eb;border-radius:8px 0 0 8px;margin-right:12px;"></div>'
                    '<div style="padding:8px 0;">'
                    f'<div style="font-size:1.1rem;color:#444;">Total Defensemen (entries)</div>'
                    f'<div style="font-size:1.7rem;font-weight:600;color:#1a202c;">{df.shape[0]}</div>'
                    '</div></div>', unsafe_allow_html=True)
    with metric_col2:
        st.markdown('<div style="display:flex;align-items:center;background:#fff;border-radius:8px;box-shadow:0 1px 4px #e5e7eb;margin-bottom:8px;">'
                    '<div style="width:8px;height:48px;background:#e5e7eb;border-radius:8px 0 0 8px;margin-right:12px;"></div>'
                    '<div style="padding:8px 0;">'
                    f'<div style="font-size:1.1rem;color:#444;">Total Features</div>'
                    f'<div style="font-size:1.7rem;font-weight:600;color:#1a202c;">{df.shape[1]}</div>'
                    '</div></div>', unsafe_allow_html=True)
    with metric_col3:
        st.markdown('<div style="display:flex;align-items:center;background:#fff;border-radius:8px;box-shadow:0 1px 4px #e5e7eb;margin-bottom:8px;">'
                    '<div style="width:8px;height:48px;background:#e5e7eb;border-radius:8px 0 0 8px;margin-right:12px;"></div>'
                    '<div style="padding:8px 0;">'
                    f'<div style="font-size:1.1rem;color:#444;">Total Missing Values</div>'
                    f'<div style="font-size:1.7rem;font-weight:600;color:#1a202c;">{df.isnull().sum().sum()}</div>'
                    '</div></div>', unsafe_allow_html=True)

    st.subheader("Original Datasets")
    bio = pd.read_excel('Bio.xlsx')
    hockey = pd.read_excel('SS.xlsx')
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bio Stats** `(Bio.xlsx)`")
        st.dataframe(bio, use_container_width=True, height=320)
    with col2:
        st.markdown("**Season Stats** `(SS.xlsx)`")
        st.dataframe(hockey, use_container_width=True, height=320)

    # Dropdown for raw data info
    with st.expander("**About The Raw Data**", expanded=False):
        st.markdown("""
        - [Bio Stats](https://www.nhl.com/stats/skaters?report=bios&reportType=season&seasonFrom=20242025&seasonTo=20242025&gameType=2&position=D&sort=a_skaterFullName&page=0&pageSize=100) & [Season Stats](https://www.nhl.com/stats/skaters?reportType=season&seasonFrom=20242025&seasonTo=20242025&gameType=2&position=D&sort=skaterFullName&page=2&pageSize=100) pulled from _nhl.com/stats_ from the 2024-25 regular season
        - Much of the missing data comes from `FOW%` and the categories associated with being drafted into the NHL (`Draft Yr`, `Round`, `Overall`), we will cover this more on the Missing Values page
        - **Bio Stats**: 
            - `S/C` - Skater Shoots (Left or Right)
            - `S/P` - Player Birth State/Provence (US/CAN)
            - `Ht` - Height (inches)
            - `Wt` - Weight (lbs)
            - `HOF` - Yes/No if player is in the Hall of Fame
            - `GP` - Games Played
            - `G` - Goals
            - `A` - Assists
            - `P` - Points (Goals + Assists)
                    
        - **Season Stats:**
            - `+/-` - Plus/Minus (goal differential)
            - `PIM` - Penalty Minutes
            - `P/GP` - Points per Game Played
            - `EVG` - Even Strength Goals
            - `EVP` - Even Strength Points
            - `PPG` - Power Play Goals
            - `PPP` - Power Play Points
            - `SHG` - Shorthanded Goals
            - `SHP` - Shorthanded Points
            - `OTG`	- Overtime Goals
            - `GWG` - Game Winning Goals
            - `S` - Shots on Goal
            - `S%` - Shooting Percentage
            - `TOI/GP` - Time On Ice per Game Played
            - `FOW%` - Face Off Win Percentage
        """)

    # Small legend table showing the encodings applied to the Combined Dataset view
    enc_info = {
        'Feature': ['S/C', 'Div', 'Conf'],
        'Encoding': [
            'L → 0, R → 1',
            'MET → 1, ATL → 2, CEN → 3, PAC → 4',
            'EC → 0, WC → 1'
        ]
    }
    enc_df = pd.DataFrame(enc_info)
    st.markdown("**Encoding Legend**")
    st.table(enc_df)

    st.subheader("Combined Dataset")
    # Display a view where S/C is encoded: L -> 0, R -> 1
    # and add encoded versions of Div and Conf as Div_En and Conf_En
    view_df = df.copy()
    if 'S/C' in view_df.columns:
        view_df['S/C'] = view_df['S/C'].map({'L': 0, 'R': 1}).fillna(view_df['S/C'])

    # Encode divisions to numbers 1-4 (ATL, MET, CEN, PAC) without removing original
    div_map = {'MET': 1, 'ATL': 2, 'CEN': 3, 'PAC': 4}
    if 'Div' in view_df.columns:
        view_df['Div_En'] = view_df['Div'].map(div_map)

    # Encode conferences: EC -> 0, WC -> 1
    conf_map = {'EC': 0, 'WC': 1}
    if 'Conf' in view_df.columns:
        view_df['Conf_En'] = view_df['Conf'].map(conf_map)

    st.dataframe(view_df, use_container_width=True, height=320)

    # Dropdown for combined data info
    with st.expander("**How is this more useful, and what's changed?**", expanded=False):
        st.markdown("""
        - The two raw datasets have been merged to be used for analysis.
        - **Taken from Bio Stats:** `Ctry`, `Ht`, `Wt`, `Draft Yr`, `Round`, and `Overall`
        - **Taken from Season Stats:** All features except for `Season`, `Pos`, and `S%`
        - Additional features such as `Div` (Division player is in based off of `Team`: Metro, Pacific, Atlantic, Central) and `Conf` (Conference player is in based off of `Div`: Eastern or Western), have been added for additional categorization
        - For housekeeping, players recorded with more than 2 teams were cleaned to only show the team they ended their season with
    - `S/C`, `Div`, and `Conf` have all been been encoded to numeric values for easier analysis
        """)

    st.subheader("Data Types and Summary Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(df.dtypes.rename("Type").reset_index().rename(columns={'index': 'Column'}))
    with col2:
        st.dataframe(df.describe().T)

# --- Class Imbalance ---
elif page == "Class Imbalance":
    st.header("Class Imbalance Analysis")
    # Remove Player and Team from categorical options
    categorical_cols = [col for col in df.select_dtypes(exclude=[np.number]).columns if col not in ['Player', 'Team', 'TOI/GP']]

    options = categorical_cols
    if not options:
        st.info("No categorical columns available.")
    else:
        selected_cat = st.selectbox("Select Category:", options)
        cat_counts = df[selected_cat].value_counts()
        # Charts side by side
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig = px.bar(
                x=cat_counts.index, y=cat_counts.values, color=cat_counts.index,
                labels={'x': selected_cat, 'y': 'Count'},
                title=f'{selected_cat} Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            fig = px.pie(values=cat_counts.values, names=cat_counts.index, hole=0.3,
                         title=f'{selected_cat} Proportion')
            st.plotly_chart(fig, use_container_width=True)
    st.write("Click on the dropdown below that dives more into the category chosen above in the graphs")
    with st.expander("**`S/C` - Skater Shoots**", expanded=False):
        st.markdown("""
        A little bit more of the Defensemen in the league shoot left than right, this doesn't nessicariarly impact how many points they get in the season, rather just a fun category to see if there are any differences between those who favor to shoot left or right.
        """)   
    with st.expander("**`Ctry` - Player's Represented Country**", expanded=False):
        st.markdown("""
        Here we can see more clearly that the majority of Defensemen in the league come from the US or Canada, closly followed up by Sweden and Russia. We will most likely see that the top producing Defensemen will come from those top 3-5 countries rather than countries represented with a smaller number of Defensemen.
        - I.E. Defensemen that perform the best will most likely come from a small subset of countries, instead of being spread across the different countries represented in the NHL.
        """)   
    with st.expander("**`Div` & `Conf` - Player's Division & Conference**", expanded=False):
        st.markdown("""
        These values are the most evenly distributed, as they are based off of the team the player is on. There are 16 teams in each division, and 8 teams in each conference, so we would expect to see a more even distribution here.
        - We may be able to see what parts of the NHL produce the best Defensemen by categorizing by Division and Conference. But we expect it to even out due to the nature of how the league is split up.
        """)   


# --- Missing Values ---
elif page == "Missing Values":
    st.header("Missing Values Analysis")

    tab1, tab2 = st.tabs(["Missing Values Overview", "Imputation"])

    # --- TAB 1: Existing Missing Values Visualization ---
    with tab1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(min(12, len(df.columns)*0.6), 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="Blues", yticklabels=False)
        plt.xlabel("Features")
        plt.title("Missing Values Heatmap")
        st.pyplot(plt.gcf())

        with st.expander("**There is a reason as to why we're missing values...**", expanded=False):
            st.markdown("""
            - Faceoffs are generally not taken by defensemen, rather forwards  
               - They will only take the faceoff if the referee has waived off all the forwards on the ice.
            - It is not mandatory to be drafted by the NHL to play in the league.  
               - There will always be a handful of players who come into the league straight out of College or from a different hockey league.
            """)

    # --- TAB 2: Imputation ---
    with tab2:
        st.subheader("Imputation of Missing Values - Filled with Column Averages")
        df_imputed = df.copy()

        # Columns to impute
        cols_to_impute = ['Draft Yr', 'Round', 'Overall']
        for col in cols_to_impute:
            if col in df_imputed.columns:
                mean_val = df_imputed[col].mean()
                df_imputed[col] = df_imputed[col].fillna(mean_val)

        # --- Show correlation heatmaps before and after imputation ---
        numeric_df = df.select_dtypes(include=[np.number])
        corr_orig = numeric_df.corr()
        mask = np.triu(np.ones_like(corr_orig, dtype=bool))

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr_orig, mask=mask, cmap='RdBu_r', center=0,
            annot=False, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax
        )
        ax.set_title("Original Combined Data - Correlation (Lower Triangle)")
        st.pyplot(fig)

        numeric_df_imp = df_imputed.select_dtypes(include=[np.number])
        corr_imp = numeric_df_imp.corr()
        mask_imp = np.triu(np.ones_like(corr_imp, dtype=bool))

        fig2, ax2 = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr_imp, mask=mask_imp, cmap='RdBu_r', center=0,
            annot=False, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax2
        )
        ax2.set_title("Imputed Combined Data - Correlation (Lower Triangle)")
        st.pyplot(fig2)

        with st.expander("**What changed after imputation?**", expanded=False):
            st.markdown("""
            The reason it doesn’t look like much has changed after imputation is because, quite frankly, not much has. Sports data, especially draft and performance data, is complex and often doen't bode well with general statistical fixes. If we could accurately “fill in the blanks” for missing sports data, the entire sports gambling industry would be in shambles.

            When we filled the missing values in `Draft Yr`, `Round`, and `Overall` with their column averages, we basically smoothed out any extremes. However, in reality, there’s a wide distribution of players drafted across all rounds (1–7), and averaging these numbers flattens that diversity.  
            
            So overall, this means the imputed values aren't adding any new information, it's just “evening the playing field” for analysis purposes, helping us keep the data complete for modeling without introducing strong bias.
            """)
        with st.expander("**Another Imputation Technique: Ignoring the missing values**", expanded=False):
            st.markdown("""
            In terms of my overall semester project, having missing values for `Draft Yr`, `Round`, `Overall`, and `FOW%` isn’t a major concern. These features don’t strongly correlate with predicting the number of points a defenseman earns in a season, especially when compared to numerical variables like Assists and Even Strength Points that have a much more direct impact on performance.

            For the purposes of my analysis, I’ll be using a much simpler imputation strategy: _ignoring these columns altogether_. Since they contribute little value to the overall narrative of my project, removing them allows me to focus on the features that truly drive on-ice production and point prediction.
            """)

# --- Correlation Analysis ---
elif page == "Correlation Analysis":
    st.header("Correlation Analysis")
    numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    import matplotlib.pyplot as plt
    # Make cells rectangular: wider figure, not square
    fig, ax = plt.subplots(figsize=(max(14, len(corr.columns)*1.2), 6))
    fig, ax = plt.subplots(figsize=(max(14, len(corr.columns)*1.2), 12))  # Double the height from 6 to 12
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=False,
        linewidths=.5,
        cbar_kws={"shrink": .8},
        annot_kws={"size": 12},
        ax=ax
    )
    plt.title('Correlation Matrix (Lower Triangle)')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    st.pyplot(fig)
    with st.expander("**Why do we need Correlation Analysis?**", expanded=False):
        st.markdown("""
        This heatmap helps I as the analyst identify strong relationships between the number of Points (`P`) a player records in a season; so that for my final project of the semester, I am able to run regression and prediction tests using the right features that have a strong correlation with each other
        """)
    with st.expander("**What do we see here that's helpful?**", expanded=False):
        st.markdown("""
        As we are using Points (`P`) as our target variable, we can see that the features that correlate the most are Assists (`A`) and Even Strength Points (`EVP`).
        
        I will continue to keep these features in mind for my final project when it comes to my regression and prediction Tests
        - Assists are potentially highly correlated with Points as playing on Defense is not the top position to score the most goals. Rather they are the ones that help to set up most of the goals shot in by a Forward. Hense why many of their Points come from Assists
        - In addition, it would also make sense as to why there is a high correlation between Points and Even Strength Points as any time a team is up one skater (on the power play) or down one skater (on the penality kill), the Defenseman is spending more time making sure other players don't interfere with the Offencemen and their scoring channces
            - A Defenseman works best on getting Points when the whole team is on the ice, working together
        """)


# --- Scatter Plots ---
elif page == "Scatter Plots":
    st.header("Scatter Plots")
    st.write("Explore Relationships Between Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2 = st.tabs(["Interactive Scatter Plot", "Pairplot"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Select X-axis:", numeric_cols, key="scatter_x")
        with col2:
            y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")

        allowed_color_cols = [col for col in ['S/C', 'Ctry', 'Div', 'Conf', 'Team'] if col in df.columns]
        color_by = st.selectbox("Color by (optional):", [None] + allowed_color_cols, key="scatter_color")
        hover_cols = [col for col in ['Player', 'Team', 'GP', 'P', 'Ctry', 'Div', 'Conf'] if col in df.columns]
        fig = px.scatter(
            df, x=x_axis, y=y_axis, color=color_by,
            title=f"{y_axis} vs {x_axis}", hover_data=hover_cols
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("**What to do here?**", expanded=False):
            st.markdown("""
            This scatter plot allows us to visually explore relationships between different numeric features in the dataset. By selecting different x and y axes, we can see trends, clusters, or outliers that may not be obvious in summary statistics alone. 
                        
            In addition- coloring by categorical variables helps to see how different groups behave in relation to the selected features.
            """)
        with st.expander("**What is Class Imbalance doing here?** (_Color by_ tab)", expanded=False):
            st.markdown("""
            Earlier we discussed how class imbalance can impact our analysis. In this scatter plot, if we color by a categorical variable that is imbalanced (`Ctry`, `S/C`, `Div`, `Conf`), we see that certain groups dominate the higher points of the plots.

            We wondered if top producing defencemen would all come from a certain class, but as we work through different features against eachother and color them by class- we come to see that at the top of the charts there's always a good bland of defencemen from different Divisions, Confrences, and Shooting Habits
            - However, in terms of `Ctry`, we do see that the top producing players exclusively only hail from the US, Canada, and Sweden. We were incorrect about assuming there would be more Russians at the top of the chart!
            """)

    with tab2:
        st.info("Below are all the possible combinations of features against each other. This tab is here to give a an overall view of how all the numeric features in the dataset relate to each other.")
        st.subheader("Pairplot (Lower Triangle)")
        import matplotlib.pyplot as plt
        import seaborn as sns
        pairplot_cols = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
        if len(pairplot_cols) > 1:
            def lower_triangle_pairplot(data, vars):
                g = sns.pairplot(
                    data[vars],
                    corner=True,
                    plot_kws={'alpha':0.7, 's':20, 'color':'dodgerblue'},
                    diag_kws={'color':'dodgerblue', 'edgecolor':'black'}
                )
                return g
            g = lower_triangle_pairplot(df, pairplot_cols)
            st.pyplot(g.fig)
        else:
            st.info("Not enough numeric columns for a pairplot.")
# --- Wrapping Up Page ---
elif page == "Wrapping Up":
    st.header("Wrapping Up")
    st.info("**Congratulations!** We have completed an interactive EDA of NHL defensemen data. Here is a summary of what was accomplished in this app:")
    st.markdown("""
    - Explored and cleaned two raw datasets (Bio and Season Stats) for NHL defensemen
    - Investigated missing values and their impact on analysis
    - Examined class imbalance and its effect on feature distributions
    - Analyzed feature correlations, focusing on what drives point production
    - Used interactive scatter plots and pairplots to visualize relationships
    - Drew insights about which features matter most for predicting points
    
    **Final Thoughts:**
    - Assists and Even Strength Points are the most important features for predicting Points among defensemen
    - Top producing defensemen come from a mix of divisions and conferences, but the highest scorers are mostly from the US, Canada, and Sweden
    - My data is now ready for regression and prediction modeling for the final project!
    """)
    st.info("Thank you for exploring my data! I hope for this to be a valuable part of my work towards Defencemen point predictions.")

   
   
    # --- Final project data preparation page ---
elif page == "Data Preperation + Collection":
    st.header("Data Preperation + Collection")
    st.info("With the addition of **[Miscellaneous Stats](https://www.nhl.com/stats/skaters?report=realtime&reportType=season&seasonFrom=20242025&seasonTo=20242025&gameType=2&position=D&sort=a_skaterFullName&page=0&pageSize=100)** from _nhl.com/stats_, we now have _**three distinct datasets**_, along with `Bio.xlsx` and `SS.xlsx` which were used in the miderm.")
    st.markdown("---")

    # Read Misc.xlsx and attach columns
    try:
        misc = pd.read_excel('Misc.xlsx')
    except Exception as e:
        st.error(f"Could not read `Misc.xlsx`: {e}")
        st.stop()

    add_cols = ['Hits', 'BkS', 'GvA', 'TkA']
    present = [c for c in add_cols if c in misc.columns]
    missing_cols = [c for c in add_cols if c not in misc.columns]

    if missing_cols:
        st.warning(f"The following columns were not found in Misc.xlsx and will be filled with NaN: {', '.join(missing_cols)}")

    # Create a new dataframe by copying the original combined df and attaching misc columns
    df_final = df.copy()
    for c in add_cols:
        if c in misc.columns:
            # align by index — assume same row order; otherwise user should merge on an ID
            df_final[c] = misc[c].values
        else:
            df_final[c] = np.nan

    # Show original combined dataset (left) and misc data (right) side-by-side
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("**Original Combined Dataset**")
        st.dataframe(df, use_container_width=True, height=350)
    with col_right:
        st.markdown("**`Misc.xlsx` - Third dataset we are pulling from**")
        st.dataframe(misc, use_container_width=True, height=350)
   
    with st.expander("**What will be added?**", expanded=False):
        st.markdown("From `Misc.xlsx`, we added the following columns to our final combined dataset: \n- `Hits` - Number of hits delivered by the defenseman\n- `BkS` - Blocked Shots\n- `GvA` - Giveaways\n- `TkA` - Takeaway")
    st.subheader("New Combined Dataset - with Misc columns")
    st.dataframe(df_final, use_container_width=True, height=420)
   
    # Show Data Types and Summary Stats for newly added Misc columns only
    st.subheader("Data Types & Summary Stats - for newly added columns")
    cols_present = [c for c in add_cols if c in df_final.columns]
    if cols_present:
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.dataframe(df_final[cols_present].dtypes.rename("Type").reset_index().rename(columns={'index': 'Column'}))
        with stats_col2:
            st.dataframe(df_final[cols_present].describe().T)
    st.info("""
### **Advanced Data Cleaning & Preprocessing**

- **Advanced Cleaning & Preprocessing:**  
  - Removed obvious nulls and standardized missing-value handling  
  - Removed duplicate team entries (kept season-ending team)  
  - Converted types where needed and aligned indices before integration  
  - Performed small manual fixes in spreadsheets where necessary  
""")
    st.info("""
### **Complex Data Integration Techniques**

- **Data consolidation:** combined `SS.xlsx`, `Bio.xlsx`, and `Misc.xlsx` into a single dataset for modeling and analysis.  
- **Data federation (practical):** because the cleaned and combined data are available inside this app, users can download the prepared dataset directly from here instead of returning to the NHL website to re-download cleaned versions. If users prefer the original raw files, they can still visit *nhl.com/stats* to obtain raw exports and perform their own cleaning.  
- **Manual data integration:** because raw tables came from *nhl.com/stats* and required cleanup, some integration steps (null fixes, name cleanup) were completed manually in Excel prior to merging, mirroring real project workflows.  
""")


# --- EDA and Visualization (Final) ---
elif page == "EDA and Visulization":
    st.header("EDA and Visulization")
    st.info("""
This page adds **two advanced visualizations**, bringing the total number of visuals in the app to be _**5+**_, and expanding the depth of analysis regarding this data.
""")

    # Rebuild final combined dataset (same logic as Data Preperation page)
    try:
        misc = pd.read_excel('Misc.xlsx')
    except Exception as e:
        st.error(f"Could not read `Misc.xlsx`: {e}")
        st.stop()

    add_cols = ['Hits', 'BkS', 'GvA', 'TkA']
    df_final = df.copy()
    for c in add_cols:
        if c in misc.columns:
            df_final[c] = misc[c].values
        else:
            df_final[c] = np.nan

    # Prepare numeric columns for analysis
    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()

    tab1, tab2 = st.tabs(["Parallel Coordinates", "Clustered Heatmap"])

    # --- Tab 1: Parallel Coordinates ---
    with tab1:
        st.subheader("Parallel Coordinates (multivariate)")
        default_cols = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        sel_cols = st.multiselect("Select numeric features to show:", numeric_cols, default=default_cols)
        # Color will be the first numeric feature chosen by the user
        if sel_cols:
            pc_df = df_final[sel_cols].copy()
            pc_df = pc_df.fillna(pc_df.mean())
            try:
                color_by = sel_cols[0]
                fig_pc = px.parallel_coordinates(pc_df, dimensions=sel_cols, color=color_by)
                # allow opacity and a colorbar by updating traces
                fig_pc.update_traces(line=dict(color=pc_df[color_by], colorscale='Viridis', showscale=True, colorbar=dict(title=color_by)))
                st.plotly_chart(fig_pc, use_container_width=True)

                # Statistical summary for visible data
                try:
                    rows_shown = pc_df.shape[0]
                    stats = pc_df[sel_cols].describe().T
                    corrs = pc_df[sel_cols].corr()
                    color_stats = pc_df[color_by].agg(['mean', 'median', 'std', 'skew']).to_dict()
                    info_text = (
                        f"**Coloring feature:** `{color_by}` mean: {color_stats['mean']:.2f}, median: {color_stats['median']:.2f}, std: {color_stats['std']:.2f}, skew: {color_stats['skew']:.2f}\n\n"
                        "**Quick notes:**\n"
                        "- Use the axis sliders above to filter ranges; the plot updates accordingly.\n"
                        "- High absolute correlations (|r| > 0.7) among selected features may indicate redundancy for modeling.\n"
                    )
                    st.info(info_text)
                    # Show descriptive stats and correlation heatmap side-by-side
                    stats_col, heat_col = st.columns([1, 1])
                    with stats_col:
                        st.markdown("**Detailed descriptive statistics (selected features)**")
                        st.dataframe(stats)
                    with heat_col:
                        st.markdown("**Correlation matrix (selected features)**")
                        try:
                            import matplotlib.pyplot as plt
                            # Plot an annotated heatmap of the correlation matrix
                            fig, ax = plt.subplots(figsize=(max(6, len(corrs.columns)*0.6), max(4, len(corrs.columns)*0.6)))
                            sns.heatmap(corrs, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
                            plt.xticks(rotation=45, ha='right')
                            plt.yticks(rotation=0)
                            fig.tight_layout()
                            st.pyplot(fig)
                        except Exception:
                            # fallback to table if plotting fails
                            st.dataframe(corrs)
                except Exception:
                    st.info("Could not compute summary statistics for the current selection.")
            except Exception as e:
                st.error(f"Could not render parallel coordinates: {e}")
        else:
            st.info("Select at least one numeric feature to render parallel coordinates.")
    
    # --- Tab 2: Clustered Heatmap ---
    with tab2:
        st.subheader("Clustered Heatmap (final combined dataset)")
        try:
            import matplotlib.pyplot as plt
            data_num = df_final.select_dtypes(include=[np.number])
            if data_num.shape[1] < 2:
                st.info("Not enough numeric features to compute a clustered heatmap.")
            else:
                corr_df = data_num.corr()
                # Use clustermap to cluster variables based on correlation
                g = sns.clustermap(corr_df.fillna(0), cmap='RdBu_r', center=0, figsize=(10, 10))
                st.pyplot(g.fig)

                # Statistical summary for clustered heatmap
                try:
                    n_features = data_num.shape[1]
                    abs_corr = corr_df.abs()
                    iu = np.triu_indices_from(abs_corr, k=1)
                    vals = abs_corr.values[iu]
                    avg_abs = float(vals.mean()) if len(vals) > 0 else 0.0
                    # Build list of pairs (i<j)
                    pairs = []
                    for i, j in zip(iu[0], iu[1]):
                        pairs.append((corr_df.index[i], corr_df.columns[j], corr_df.iloc[i, j]))
                    pairs_sorted = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
                    high_pairs = [p for p in pairs_sorted if abs(p[2]) >= 0.7]
                    info_text = (
                        f"**Numeric features used:** {n_features}\n\n"
                        f"**Highly correlated pairs (|r| >= 0.7):** {len(high_pairs)}\n"
                    )
                    st.info(info_text)
                    if len(high_pairs) > 0:
                        top_display = pd.DataFrame(high_pairs[:10], columns=['Feature 1', 'Feature 2', 'Correlation'])
                        st.markdown("**Top highly correlated pairs (up to 10):**")
                        st.dataframe(top_display)
                    st.markdown("**Full correlation matrix used for clustering**")
                    st.dataframe(corr_df)
                except Exception:
                    st.info("Could not compute summary statistics for the clustered heatmap.")
        except Exception as e:
            st.error(f"Could not render clustered heatmap: {e}")


# --- Model Development and Evaluation ---
elif page == "Model Development and Evaluation":
    st.header("Model Development and Evaluation")
    st.info("This section implements **two machine learning models** and includes **evaluation and comparison** to assess performance and interpret results.")

    # Rebuild final combined dataset
    try:
        misc = pd.read_excel('Misc.xlsx')
    except Exception as e:
        st.error(f"Could not read `Misc.xlsx`: {e}")
        st.stop()

    add_cols = ['Hits', 'BkS', 'GvA', 'TkA']
    df_final = df.copy()
    for c in add_cols:
        if c in misc.columns:
            df_final[c] = misc[c].values
        else:
            df_final[c] = np.nan

    st.subheader("Snapshot of Data")
    st.dataframe(df_final.head(8), use_container_width=True)

    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns available for modeling.")
        st.stop()

    with st.expander("Modeling inputs & preprocessing", expanded=True):
        target_default = 'P' if 'P' in numeric_cols else numeric_cols[-1]
        target = st.selectbox("Select target variable:", numeric_cols, index=numeric_cols.index(target_default) if target_default in numeric_cols else 0)
        features = st.multiselect("Select feature columns (numeric):", [c for c in numeric_cols if c != target], default=[c for c in ['A','GP'] if c in numeric_cols][:4])
        test_size = st.slider("Test set proportion:", 0.05, 0.5, 0.2)
        do_scale = st.checkbox("Standardize features (Z-score)", value=True)

    # Use a fixed random state for reproducibility (removed interactive input)
    random_state = 42

    if not features:
        st.info("Choose at least one feature to train models.")
    else:
        # Build X, y
        X = df_final[features].copy()
        y = df_final[target].copy()
        # Drop rows with missing target or all-NaN features
        mask = ~y.isnull() & X.notnull().all(axis=1)
        X = X[mask]
        y = y[mask]

        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

        # scaling
        if do_scale:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        else:
            X_train_scaled, X_test_scaled = X_train, X_test

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=int(random_state))
        }

        results = []
        preds = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                preds[name] = y_pred
                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                # cross-val on the training data
                try:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    cv_mean = float(np.mean(cv_scores))
                except Exception:
                    cv_mean = None
                results.append({'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CV_R2_mean': cv_mean})
            except Exception as e:
                results.append({'Model': name, 'RMSE': None, 'MAE': None, 'R2': None, 'CV_R2_mean': None})

        res_df = pd.DataFrame(results).set_index('Model')
        st.subheader("Model Comparison")
        st.dataframe(res_df)

        # Info boxes explaining selection and validation
        st.info("**Model selection summary**: _Linear Regression_ is a simple baseline linear model while _Decision Trees_ capture non-linear interactions but can overfit. \n - We compare using RMSE/MAE/R2 on a held-out test set and cross-validate R2 on the training data to estimate generalization.")
        with st.expander("Validation techniques and model selection notes", expanded=False):
            st.markdown("""
            - We split our data into training/testing groups to get an unbiased estimate of how it will perform.
            - Cross-validation (k-fold) on the training data helps choose parameters while reducing overfitting.
            - Generally we preferr simpler models if performance is similar 
                - Meanwhile, decision Trees often require pruning or max-depth tuning.
            - Below we will be able to check residual plots and distribution of errors to spot bias.
            """)

        # Residual plots and diagnostics
        st.subheader("Residuals and Predictions")
        for name, y_pred in preds.items():
            try:
                resid = y_test - y_pred
                st.markdown(f"**{name}**")
                # Prepare the two diagnostic plots
                fig_scatter = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'}, title=f"{name} - Actual vs Predicted")
                fig_scatter.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(dash='dash'))

                fig_res = px.histogram(resid, nbins=30, title=f"{name} - Residuals distribution")

                # Display the two plots side-by-side
                left_col, right_col = st.columns(2)
                with left_col:
                    st.plotly_chart(fig_scatter, use_container_width=True)
                with right_col:
                    st.plotly_chart(fig_res, use_container_width=True)
            except Exception:
                st.info(f"Could not produce diagnostics for {name}.")

        # Offer predictions download
        out_df = X_test.copy()
        out_df['Actual_'+str(target)] = y_test
        for name, y_pred in preds.items():
            out_df['Pred_'+name.replace(' ', '_')] = y_pred
        csv = out_df.to_csv(index=False)
        st.download_button("Download test-set predictions (CSV)", data=csv, file_name='model_predictions.csv', mime='text/csv')


# --- Data Processing and Feature Engineering ---
elif page == "Data Processing and Feature Engineering":
    st.header("Data Processing and Feature Engineering")
    st.info("Two multiple enginering techniques have been implemented on this page: _**Numeric Transoformations**_ and _**Categorical Encoding**_.")

    # Rebuild final combined dataset (same logic as Data Preperation page)
    try:
        misc = pd.read_excel('Misc.xlsx')
    except Exception as e:
        st.error(f"Could not read `Misc.xlsx`: {e}")
        st.stop()

    add_cols = ['Hits', 'BkS', 'GvA', 'TkA']
    df_final = df.copy()
    for c in add_cols:
        if c in misc.columns:
            df_final[c] = misc[c].values
        else:
            df_final[c] = np.nan

    st.subheader("Snapshot of Data")
    st.dataframe(df_final.head(10), use_container_width=True)

    numeric_cols = df_final.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found to process.")
        st.stop()

    st.markdown("**Feature Engineering Options**")
    tab_num, tab_cat = st.tabs(["Numeric Transforms", "Categorical Encoding"])

    # --- Numeric transforms tab (existing behavior) ---
    with tab_num:
        col1, col2 = st.columns(2)
        with col1:
            log_cols = st.multiselect("Log-transform columns (adds `_log`):", numeric_cols)
        with col2:
            z_cols = st.multiselect("Standardize columns (adds `_z`):", numeric_cols)

        custom_name = st.text_input("Optional name for a simple ratio feature (A/B -> name)", value="", key='ratio_name')
        ratio_a = None
        ratio_b = None
        if custom_name:
            ratio_a = st.selectbox("Numerator (A)", [None] + numeric_cols, key='ratio_a')
            ratio_b = st.selectbox("Denominator (B)", [None] + numeric_cols, key='ratio_b')

        if st.button("Apply transforms and preview", key='apply_numeric'):
            df_proc = df_final.copy()
            # apply log transforms
            for c in log_cols:
                try:
                    df_proc[c + '_log'] = np.log1p(df_proc[c])
                except Exception:
                    df_proc[c + '_log'] = np.nan
            # apply z-score standardization
            for c in z_cols:
                try:
                    df_proc[c + '_z'] = (df_proc[c] - df_proc[c].mean()) / df_proc[c].std()
                except Exception:
                    df_proc[c + '_z'] = np.nan
            # simple ratio feature
            if custom_name and ratio_a and ratio_b and ratio_a in df_proc.columns and ratio_b in df_proc.columns:
                with np.errstate(divide='ignore', invalid='ignore'):
                    df_proc[custom_name] = df_proc[ratio_a] / df_proc[ratio_b]

            st.success("Transforms applied — preview below (first 10 rows)")
            st.dataframe(df_proc.head(10), use_container_width=True)

            st.markdown("**Descriptive stats for newly created features**")
            new_cols = [c for c in df_proc.columns if c.endswith('_log') or c.endswith('_z') or c == custom_name]
            if new_cols:
                st.dataframe(df_proc[new_cols].describe().T)
            else:
                st.info("No new features were created.")

            # provide download
            csv = df_proc.to_csv(index=False)
            st.download_button("Download processed CSV", data=csv, file_name='df_processed.csv', mime='text/csv')

    # --- Categorical encoding tab ---
    with tab_cat:
        st.subheader("One-hot encode categorical variables")
        # Candidate categorical columns for encoding
        available_cat = [c for c in ['S/C', 'Conf', 'Div'] if c in df_final.columns]
        if not available_cat:
            st.info("No categorical columns (`S/C`, `Conf`, `Div`) found in the dataset.")
        else:
            enc_select = st.multiselect("Select columns to one-hot encode:", available_cat)
            drop_first_enc = st.checkbox("Drop first level (avoid multicollinearity)", value=True)
            dummy_na = st.checkbox("Create indicator for NaNs (dummy_na)", value=False)
            drop_originals = st.checkbox("Drop original categorical columns after encoding", value=False)

            if st.button("Apply encoding and preview", key='apply_enc'):
                df_enc = df_final.copy()
                if not enc_select:
                    st.info("Select at least one column to encode.")
                else:
                    try:
                        # Use get_dummies for selected columns only
                        prefixes = [c.replace('/', '_') for c in enc_select]
                        dummies = pd.get_dummies(df_enc[enc_select], prefix=prefixes, prefix_sep='_', drop_first=drop_first_enc, dummy_na=dummy_na)
                        # concat and optionally drop originals
                        df_enc = pd.concat([df_enc, dummies], axis=1)
                        if drop_originals:
                            df_enc = df_enc.drop(columns=enc_select)

                        st.success("Encoding applied, preview below")
                        st.dataframe(df_enc.head(10), use_container_width=True)

                        # show new columns and basic counts
                        new_cols = [c for c in df_enc.columns if c not in df_final.columns]
                        if new_cols:
                            st.markdown("**Newly created one-hot columns (sample counts)**")
                            counts = df_enc[new_cols].sum().rename('Count').to_frame()
                            st.dataframe(counts)
                        else:
                            st.info("No new one-hot columns were produced.")

                        csv_enc = df_enc.to_csv(index=False)
                        st.download_button("Download encoded CSV", data=csv_enc, file_name='df_encoded.csv', mime='text/csv')
                    except Exception as e:
                        st.error(f"Failed to encode selected columns: {e}")




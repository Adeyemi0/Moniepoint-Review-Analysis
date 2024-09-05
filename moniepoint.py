import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import plotly.express as px

# Function to provide a summary of the dataset
def generate_summary(df):
    total_reviews = len(df)
    avg_rating = df['rating'].mean()
    latest_review = df['date'].max()
    
    summary = (
        f"Total Reviews: {total_reviews}\n"
        f"Average Rating: {avg_rating:.2f}\n"
        f"Most Recent Review Date: {latest_review.date()}"
    )
    return summary

# Load dataset with caching for efficiency
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Function to retrieve relevant data based on the query
def retrieve_relevant_data(query, df):
    query_keywords = query.split()  # Basic keyword extraction by splitting on spaces
    relevant_rows = df[df['review'].str.contains('|'.join(query_keywords), case=False, na=False)]
    
    if relevant_rows.empty:
        return "No relevant data found", pd.DataFrame()

    return relevant_rows[['review', 'rating', 'date']].head(5), relevant_rows

# Function to generate an answer using the language model
def get_answer(query, df):
    relevant_data_sample, relevant_data = retrieve_relevant_data(query, df)
    
    if relevant_data_sample.empty:
        return "Data not available", pd.DataFrame()

    data_string = relevant_data_sample.to_string(index=False, max_colwidth=50)
    final_prompt = question_prompt.format(query=query, data=data_string)
    response = llm.invoke(final_prompt)
    
    return response, relevant_data_sample

# Load dataset
df = load_data("moniepoint.csv")

if not df.empty:
    # Determine the min and max date for the date filter
    min_date = df['date'].min()
    max_date = df['date'].max()

    # Initialize the language model
    llm = Ollama(model="phi3")

    # Define the question prompt template
    question_prompt = PromptTemplate(
        input_variables=["query", "data"],
        template=(
            "Answer the following question based on the review dataset ONLY. "
            "Here is the relevant data: {data}. "
            "If the answer cannot be found in the data, say 'Data not available'. "
            "Question: {query}"
        )
    )

    # Streamlit App Layout
    st.title("Moniepoint Review Analysis")

    # Introduction section
    st.write("""
    ### Introduction

    Welcome to the Moniepoint Review Analysis app. This tool allows you to analyze reviews from the Google Playstore and Apple Store.

    #### Features:
    - **View Dataset Summary**: Get an overview of the review dataset, including total reviews, average rating, and the most recent review date.
    - **Visualize Ratings**: View a bar chart of rating distributions.
    - **Ask Specific Questions**: Use the text input field to ask questions about the dataset, such as inquiries on specific aspects of user feedback.

    Please enter your query below and select a date range to filter the reviews if needed. Click 'Submit' to get the answer.
    """)

    # Sidebar filters
    st.sidebar.header("Filter Options")

    # Date filter
    date_filter = st.sidebar.date_input(
        "Filter reviews by date:",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]  # Default to the full date range
    )

    # Source filter (Google Play Store / Apple Store)
    source_options = df['source'].unique().tolist()  # Get unique values from the source column
    selected_sources = st.sidebar.multiselect(
        "Filter by Source:",
        options=source_options,
        default=source_options  # Default to selecting all sources
    )

    # Apply filtering based on date and source
    df_filtered = df[
        (df['date'] >= pd.to_datetime(date_filter[0])) & 
        (df['date'] <= pd.to_datetime(date_filter[1])) &
        (df['source'].isin(selected_sources))
    ]

    # Display dataset summary and bar chart
    st.write("### Dataset Summary")
    summary = generate_summary(df_filtered)
    st.write(summary)

    fig = px.histogram(df_filtered, x="rating", nbins=10, title="Rating Distribution")
    st.plotly_chart(fig)

    # Move the query and submit button after the summary and bar chart
    st.write("### Ask a Question")

    # User input for query
    user_query = st.text_input("Enter your question about the dataset (e.g., 'What are users saying about customer support?'):")

    # Submit button to generate answer and display additional summaries
    if st.button('Submit'):
        # Provide review summary based on user query
        if user_query.strip():
            with st.spinner('Generating answer...'):
                answer, data_used = get_answer(user_query, df_filtered)
                st.write("### Answer:")
                st.write(answer)
                
                if not data_used.empty:
                    st.write("### Relevant Data Used:")
                    st.dataframe(data_used)
        else:
            st.warning("Please enter a valid question.")
else:
    st.warning("No data available to display.")

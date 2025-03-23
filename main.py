from datetime import datetime
import re
import streamlit as st
import pandas as pd
import plotly as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from streamlit_navigation_bar import st_navbar
import numpy as np
from scipy.signal import find_peaks
import requests
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_within_date_range(article_date_str, start_date, end_date):
            """
            Checks if an article's published date (in ISO 8601 format) falls within the start_date and end_date.
            """
            try:
                article_date = datetime.fromisoformat(article_date_str.replace("Z", "+00:00")).date()
                return start_date <= article_date <= end_date
            except Exception:
                return False

def fetch_news_from_mediastack(query, maxima_date):
  
   
    maxima_date_obj = datetime.strptime(maxima_date, "%Y-%m-%d").date()
    start_date_obj = maxima_date_obj - timedelta(days=5)
    
   
    start_date_str = start_date_obj.strftime("%Y-%m-%d")
    end_date_str = maxima_date_obj.strftime("%Y-%m-%d")
    
    url = "http://api.mediastack.com/v1/news"
    params = {
        "access_key": '19f51ea24a110a574bfdbb67f29a5a91',
        "keywords": query,
        "languages": "en",
        "limit": 20,  
        "date": f"{start_date_str},{end_date_str}"
    }
    
    response = requests.get(url, params=params)
    print(params)
    data = response.json()
    # print('res',response)
    news_articles = []
    
    if "data" in data:
        for article in data["data"]:
            published_at = article.get("published_at", "")
            # Filter: include only articles published within the desired range.
            if not is_within_date_range(published_at, start_date_obj, maxima_date_obj):
                continue
            news_articles.append({
                "title": article.get("title", ""),
                "link": article.get("url", ""),
                "published": published_at,
                "source": article.get("source", ""),
                "description": article.get("description", "")
            })
    return news_articles
st.markdown("### Upload your CSV file")


uploaded_files = st.file_uploader(
    "Choose a CSV file", type=["csv"], accept_multiple_files=True
)

time_grouping = st.radio(
    "Choose the time grouping for the tweet timeline:",
    ('Daily', 'Hourly')
)
time_grouping_user = st.radio(
    "Choose the time grouping for the user creation timeline:",
    ('Daily', 'Quarterly')
)

for uploaded_file in uploaded_files:
    
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        dataframe = pd.read_csv(stringio,dtype={'tweet_id': str})
        yt = dataframe
        expected_columns = [
    'tweet_id', 'screen_name', 'text', 'created_at', 'user_screen_name', 
    'user_name', 'user_description', 'user_followers_count', 'user_favourites_count', 
    'user_avatar', 'user_verified', 'user_friends_count', 'user_creation_date', 
    'user_blue_verified', 'user_id_str', 'user_profile_banner_url', 
    'user_profile_image_url', 'user_status_count', 'conversation_id_str', 
    'hashtags', 'urls', 'user_mentions', 'in_reply_to_screen_name', 
    'in_reply_to_status_id_str', 'in_reply_to_user_id_str', 'is_quote_status', 
    'quote_count', 'reply_count', 'retweet_count', 'retweeted', 'tweet_user_id_str',
    'view Count'
]

        if not all(column in dataframe.columns for column in expected_columns):
            st.error("Invalid CSV file. Please ensure the file contains the correct columns.")
            break
    
        

        

        integer_columns = dataframe.select_dtypes(include=['int64', 'int32']).columns

    
        dataframe['created_at'] = pd.to_datetime(dataframe['created_at'], errors='coerce')
        dataframe['user_creation_date'] = pd.to_datetime(dataframe['user_creation_date'], errors='coerce')

      
        dataframe = dataframe.dropna(subset=['created_at', 'user_creation_date'])
        new_df = dataframe.dropna(subset=['created_at', 'user_creation_date'])
        # st.write(len(new_df))
        unique_users_df = dataframe.drop_duplicates(subset='user_name')
        
        
        if time_grouping_user == 'Daily':
            unique_users_df['user_time_group'] = unique_users_df['user_creation_date'].dt.date
        elif time_grouping_user == 'Quarterly':
            unique_users_df['user_time_group'] = unique_users_df['user_creation_date'].dt.to_period('Q').dt.to_timestamp()

        user_count = unique_users_df.groupby('user_time_group').size().reset_index(name="user_count")
        

     
        if time_grouping == 'Daily':
            dataframe['tweet_time_group'] = dataframe['created_at'].dt.date
        elif time_grouping == 'Hourly':
            dataframe['tweet_time_group'] = dataframe['created_at'].dt.strftime('%Y-%m-%d %H:00:00')

        tweet_count_by_time = dataframe.groupby('tweet_time_group').size().reset_index(name='tweet_count')

# timeline graphs 
        
        st.markdown(f"### Tweet Creation Timeline - {time_grouping} View")
        fig = go.Figure()

        tweet_max = tweet_count_by_time['tweet_count'].max()
        peak_time = tweet_count_by_time[tweet_count_by_time['tweet_count'] == tweet_max].iloc[0]

        peak_time['tweet_time_group'] = pd.to_datetime(peak_time['tweet_time_group'], errors='coerce')
        # st.write(peak_time['tweet_time_group'].strftime('%Y-%m-%d'))

        if peak_time['tweet_time_group'].tzinfo is None:
            peak_time['tweet_time_group'] = peak_time['tweet_time_group'].tz_localize('UTC')

        if tweet_count_by_time['tweet_time_group'].dtype != 'datetime64[ns, UTC]':
            tweet_count_by_time['tweet_time_group'] = pd.to_datetime(tweet_count_by_time['tweet_time_group'], errors='coerce').dt.tz_localize('UTC', ambiguous='NaT')

        fig.add_trace(go.Scatter(
            x=tweet_count_by_time['tweet_time_group'],
            y=tweet_count_by_time['tweet_count'],
            fill='tozeroy',
            mode='lines',
            line=dict(color='rgba(100, 100, 255, 0.6)', width=2),
            name="Tweet Count"
        ))

        fig.add_trace(go.Scatter(
            x=[peak_time['tweet_time_group']],
            y=[tweet_max],
            mode='markers',
            marker=dict(color='red', size=10),
            name='Global Maxima'
        ))

        local_maxima_indices, _ = find_peaks(tweet_count_by_time['tweet_count'])
        local_maxima = tweet_count_by_time.iloc[local_maxima_indices]

        # Filter local maxima excluding global maxima
        filtered_local_maxima = local_maxima[(local_maxima['tweet_count'] >= 0.5 * tweet_max) &
                                            (local_maxima['tweet_time_group'] != peak_time['tweet_time_group'])]

        for i, (_, row) in enumerate(filtered_local_maxima.iterrows()):
            color = f"hsl({i * 60 % 360}, 70%, 50%)"
            fig.add_trace(go.Scatter(
                x=[row['tweet_time_group']],
                y=[row['tweet_count']],
                mode='markers',
                marker=dict(color=color, size=8),
                name=f"Local Maxima - {row['tweet_time_group'].strftime('%Y-%m-%d')}"
            ))

        # Filter accounts created within 5 days of local maxima
        unique_users_near_all_maxima = pd.DataFrame()

        for _, row in filtered_local_maxima.iterrows():
            local_date_range_start = pd.to_datetime(row['tweet_time_group'], errors='coerce') - pd.Timedelta(days=5)
            local_date_range_end = pd.to_datetime(row['tweet_time_group'], errors='coerce')

            if local_date_range_start.tzinfo is None:
                local_date_range_start = local_date_range_start.tz_localize('UTC')
            if local_date_range_end.tzinfo is None:
                local_date_range_end = local_date_range_end.tz_localize('UTC')

            users_near_local = dataframe[(dataframe['user_creation_date'] >= local_date_range_start) &
                                        (dataframe['user_creation_date'] <= local_date_range_end)]

            unique_users_near_local = users_near_local.drop_duplicates(subset='user_name')
            unique_users_near_all_maxima = pd.concat([unique_users_near_all_maxima, unique_users_near_local], ignore_index=True)

        unique_users_near_all_maxima = unique_users_near_all_maxima.drop_duplicates(subset='user_name')

        users_len = dataframe.drop_duplicates(subset='user_name')
        len_df = len(dataframe)
        print("number of users:",len(users_len))
        print("total tweets",len_df)

        

        
        

        fig.update_layout(
            title=f"Tweet Creation Timeline ({time_grouping} View)",
            xaxis_title=f"{time_grouping} Time",
            yaxis_title="Number of Tweets",
            template='plotly_dark',
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="white"),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.3)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False
            )
        )

        st.plotly_chart(fig)

        tweet_csv = tweet_count_by_time.to_csv(index=False).encode('utf-8')
        st.download_button("Download Tweet Count Data as CSV", data=tweet_csv, file_name="tweet_count_data.csv", mime="text/csv")

        st.markdown(f"### User Creation Timeline - {time_grouping_user} View")
        fig_user = go.Figure()
        fig_user.add_trace(go.Scatter(
            x=user_count['user_time_group'],
            y=user_count['user_count'],
            fill='tozeroy',
            mode='lines',
            line=dict(color='rgba(255, 100, 100, 0.6)', width=2),
            name="User Count"
        ))

        fig_user.update_layout(
            title=f"User Creation Timeline ({time_grouping_user} View)",
            xaxis_title=f"{time_grouping_user} Time",
            yaxis_title="Number of Users",
            template='plotly_dark',
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="white"),
            xaxis=dict(
                tickformat="%Y-Q%q",
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.3)',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)',
                zeroline=False
            )
        )
        st.plotly_chart(fig_user)
        # st.write(len(unique_users_df))
        # st.write(len(dataframe))
        st.write(f"### Accounts Created within 5 Days of Global and Local Maxima")
        try:
            st.write(unique_users_near_all_maxima[['user_screen_name', 'user_creation_date', 'user_status_count']])
        except Exception as e:
            st.write("")
        st.write(f"### Local Maxima of Tweet Creation")
        try:

            st.write(filtered_local_maxima[['tweet_time_group', 'tweet_count']])
        except Exception as e:
            st.write("")
        user_csv = user_count.to_csv(index=False).encode('utf-8')
        st.download_button("Download User Count Data as CSV", data=user_csv, file_name="user_count_data.csv", mime="text/csv")

        st.divider()
        
        
        # if st.checkbox("Check for news articles related hashtag"):
        #     file_name_news = uploaded_file.name.split('.csv')[0]
            
        #     mediastack_news = fetch_news_from_mediastack(
        #         file_name_news,
        #         peak_time['tweet_time_group'].strftime('%Y-%m-%d')
        #     )
            
            
        #     all_news = mediastack_news
        #     data = []
        #     for article in all_news:
                
        #         read_more = f'<a href="{article["link"]}" target="_blank">Read More</a>'
        #         data.append({
        #             "Title": article["title"],
        #             "Source": article.get("source", "Unknown"),
        #             "Published": article["published"],
        #             "Link": read_more
        #         })
            
        #     df_news = pd.DataFrame(data)
            
        #     st.write(df_news.to_html(escape=False), unsafe_allow_html=True)


        # st.divider()

        
        dataframe['tweet_time_group'] = pd.to_datetime(dataframe['tweet_time_group'], errors='coerce')
        if not pd.api.types.is_datetime64tz_dtype(dataframe['tweet_time_group']):
            dataframe['tweet_time_group'] = dataframe['tweet_time_group'].dt.tz_localize('UTC')

        # Find the global peak time
        tweet_max = tweet_count_by_time['tweet_count'].max()
        global_peak_time = tweet_count_by_time[tweet_count_by_time['tweet_count'] == tweet_max].iloc[0]
        global_peak_time['tweet_time_group'] = pd.to_datetime(global_peak_time['tweet_time_group'], errors='coerce')

        if global_peak_time['tweet_time_group'].tzinfo is None:
            global_peak_time['tweet_time_group'] = global_peak_time['tweet_time_group'].tz_localize('UTC')

        global_end_date = global_peak_time['tweet_time_group']
        print('maxima date ',global_end_date)

        
        dataframe['user_creation_date'] = pd.to_datetime(dataframe['user_creation_date'], errors='coerce')
        if not pd.api.types.is_datetime64tz_dtype(dataframe['user_creation_date']):
            dataframe['user_creation_date'] = dataframe['user_creation_date'].dt.tz_localize('UTC')

        
        accounts_created_on_global_maxima = dataframe[
            (dataframe['user_creation_date'].dt.date == global_end_date.date()) &
            (dataframe['tweet_time_group'] == global_end_date)
        ]

        
        accounts_created_on_global_maxima['tweet_count'] = accounts_created_on_global_maxima.groupby('user_screen_name')['tweet_id'].transform('count').fillna(0)

        
        unique_accounts_on_global_maxima = accounts_created_on_global_maxima.drop_duplicates(subset='user_screen_name')

        total_tweets_in_dataset = dataframe['tweet_id'].nunique()
        total_unique_accounts_in_dataset = dataframe['user_name'].nunique()
        accounts_to_total_tweets_ratio = total_unique_accounts_in_dataset / total_tweets_in_dataset if total_tweets_in_dataset > 0 else 0

        

        unique_accounts_on_global_maxima = unique_accounts_on_global_maxima.reset_index(drop=True)
        unique_accounts_on_global_maxima.index += 1
        

        # Filter tweets for the global maxima date

        tweets_on_maxima_date = dataframe[dataframe['created_at'].dt.date == global_end_date.date()]

        # Calculate statistics for that date
        total_tweets_on_maxima_date = len(tweets_on_maxima_date)
        unique_users_on_maxima_date = tweets_on_maxima_date['user_name'].nunique()

        # Calculate the ratio
        tweets_per_user_ratio = total_tweets_on_maxima_date / unique_users_on_maxima_date if unique_users_on_maxima_date > 0 else 0

  
        # st.write(f"### Tweet Activity on Global Maxima Date ({global_end_date.date()})")
       
                
        # st.write(f'### total tweets in dataset by unique accounts: {total_tweets_in_dataset}')
        total_unique_accounts_in_dataset = dataframe['user_name'].nunique()
        ratio = total_tweets_in_dataset / total_unique_accounts_in_dataset
      
                


        
    
        fig = go.Figure()

        
        accounts_created_count = len(unique_accounts_on_global_maxima)

        
        tweets_count_on_global_maxima = unique_accounts_on_global_maxima['tweet_count'].sum()

       
        if accounts_created_count > 0:
            tweets_per_account_on_global_maxima = tweets_count_on_global_maxima / accounts_created_count
        else:
            tweets_per_account_on_global_maxima = 0

        
        total_unique_accounts_in_dataset = dataframe['user_name'].nunique()
        total_tweets_in_dataset = dataframe['tweet_id'].nunique()

       
        if total_unique_accounts_in_dataset > 0:
            tweets_per_account_ratio = total_tweets_in_dataset / total_unique_accounts_in_dataset
        else:
            tweets_per_account_ratio = 0

       
        fig.add_trace(go.Bar(
            x=['Ratio of tweets on Global Maxima to unique users', 'Ratio of Total Tweets per Unique Account'],
            y=[tweets_per_user_ratio, ratio],
            marker=dict(color='rgba(100, 100, 255, 0.6)'),
            name='Ratios'
        ))

       
        fig.update_layout(
            title="Peak trend vs Overall Trend",
            xaxis_title="Metrics",
            yaxis_title="Ratio",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="white"),
            barmode='group',
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.3)', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False)
        )

        st.plotly_chart(fig)


        
        global_maxima_on_accounts_csv = unique_accounts_on_global_maxima[['user_screen_name', 'user_creation_date', 'tweet_count']].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Accounts Data Created on Global Maxima as CSV",
            data=global_maxima_on_accounts_csv,
            file_name="global_maxima_on_accounts.csv",
            mime="text/csv"
        )



        st.divider()

        
        def extract_mentions(text):
            mentions = re.findall(r'(?<!RT\s)@\w+', text)
            return mentions

        dataframe['mentions'] = dataframe['text'].apply(lambda x: extract_mentions(x) if pd.notnull(x) else [])

        all_mentions = dataframe.explode('mentions')['mentions'].dropna()
        mention_counts = all_mentions.value_counts().reset_index()
        mention_counts.columns = ['Mention', 'Count']
        top_mentions = mention_counts.head(20)

        st.markdown("### Top 20 Most Mentioned Users (Excluding Retweets)")
        fig_mentions = go.Figure(go.Bar(
            x=top_mentions['Mention'],
            y=top_mentions['Count'],
            marker=dict(color='rgba(244, 126, 37, 0.98)')
        ))

        fig_mentions.update_layout(
            title="Top 20 Most Mentioned Users",
            xaxis_title="User Mentions",
            yaxis_title="Frequency",
            template='plotly_dark'
        )
        st.plotly_chart(fig_mentions)

        mentions_csv = mention_counts.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Mentions Data as CSV",
            data=mentions_csv,
            file_name="mentions_data.csv",
            mime="text/csv"
        )
       
        st.divider()

        user_column = 'user_screen_name' if 'user_screen_name' in new_df.columns else 'user_name'
        
        if user_column in new_df.columns:  # Check if the user column exists
            # Group by the selected user column and calculate tweet counts
            user_tweet_counts = new_df.groupby(user_column).size().reset_index(name='tweet_count')
            csv_file_name = "user_tweet_counts.csv"
            
            # Sort by tweet count in descending order to get the top users
            top_users = user_tweet_counts.sort_values(by='tweet_count', ascending=False).head(20)
            top_users.to_csv(csv_file_name, index=False)
            # Calculate total users and total tweets
            total_users = user_tweet_counts[user_column].nunique()
            total_tweets = new_df.shape[0]

            # Find the maximum tweets by a single user
            max_tweets_by_user = user_tweet_counts['tweet_count'].max()

            
            # Display the title
            st.markdown(f"### Top {len(top_users)} Users by Tweet Count")

            # Create a bar chart
            fig_users = go.Figure(go.Bar(
                x=top_users[user_column],
                y=top_users['tweet_count'],
                marker=dict(color='rgba(236, 14, 117, 0.8)'),  # Magenta shade
            ))

            # Update chart layout
            fig_users.update_layout(
                title=f"Top {len(top_users)} Users with Most Tweets",
                xaxis_title="Screen Name" if user_column == 'user_screen_name' else "Username",
                yaxis_title="Tweet Count",
                template='plotly_dark',
                xaxis=dict(
                    type='category',
                    tickmode='array',
                    tickvals=top_users[user_column],
                    ticktext=top_users[user_column]
                )
            )
            
            # Render the chart in Streamlit
            st.plotly_chart(fig_users)
            with open(csv_file_name, 'rb') as file:
                st.download_button(
                    label="Download User Tweet Counts as CSV",
                    data=file,
                    file_name=csv_file_name,
                    mime='text/csv'
                )
        else:
            st.write("No user column found in the data (neither 'user_screen_name' nor 'user_name').")
        st.divider()

        from datetime import datetime

        if 'user_status_count' in dataframe.columns and 'user_creation_date' in dataframe.columns and 'user_screen_name' in dataframe.columns and 'user_description' in dataframe.columns:
            dataframe = dataframe.drop_duplicates(subset=['user_screen_name'])
            
            # Convert to datetime and handle timezone
            dataframe['user_creation_date'] = pd.to_datetime(dataframe['user_creation_date'], errors='coerce').dt.tz_localize(None)
            
            # Calculate account age in days
            dataframe['days_since_creation'] = (datetime.now() - dataframe['user_creation_date']).dt.days
            dataframe['days_since_creation'] = dataframe['days_since_creation'].apply(lambda x: max(x, 1))
            
            # Calculate daily tweet rate
            dataframe['daily_tweets'] = dataframe['user_status_count'] / dataframe['days_since_creation']
            
           
            media_keywords = ['news', 'media', 'journalist', 'reporter', 'press', 
                            'editor', 'channel', 'tv', 'radio', 'newspaper', 
                            'magazine', 'anchor']
            
            
            def is_media_account(description):
                if isinstance(description, str):
                    description = description.lower()
                    return any(keyword in description for keyword in media_keywords)
                return False
            
            
            dataframe['is_media'] = dataframe['user_description'].apply(is_media_account)
           
            def label_bot(daily_tweets, is_media):
                if daily_tweets >= 100:
                    return 'Bot'
                elif daily_tweets >= 20 and daily_tweets < 100:
                    return 'Potential Bot'
                elif is_media:
                    return 'Potential Bot (Media)'
                else:
                    return 'Need More Analysis'
            
          
            dataframe['bot_label'] = dataframe.apply(lambda row: label_bot(row['daily_tweets'], row['is_media']), axis=1)
            
            st.write(dataframe[['user_screen_name', 'user_name', 'user_status_count', 'user_creation_date', 'daily_tweets', 'bot_label']])
        else:
            st.error("Required columns are missing from the dataframe.")
        st.divider()
        num_tweets = st.slider("Select the number of top retweeted posts to display", min_value=1, max_value=30, value=10)
     
        if 'retweet_count' in dataframe.columns and 'tweet_id' in dataframe.columns:
          
            top_retweets = dataframe[['tweet_id', 'retweet_count']].sort_values(by='retweet_count', ascending=False).head(num_tweets)
            
            
            top_retweets['tweet_id'] = top_retweets['tweet_id'].astype(str)
            
            # Create URLs for each tweet
            top_retweets['tweet_url'] = 'https://twitter.com/i/web/status/' + top_retweets['tweet_id']
            
           
            fig_retweets = go.Figure(go.Scatter(
                x=top_retweets['tweet_id'],
                y=top_retweets['retweet_count'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=top_retweets['retweet_count'], 
                    colorscale='Viridis',  
                    showscale=True
                ),
                customdata=top_retweets['tweet_url'],  
                hovertemplate="<b>Retweet Count: %{y}</b><extra></extra>"  
            ))

            
            fig_retweets.update_layout(
                title=f"Top {num_tweets} Tweets with Highest Retweet Count",
                xaxis_title="Tweet ID",
                yaxis_title="Retweet Count",
                template='plotly_dark',
                xaxis=dict(type='category', tickmode='array', tickvals=top_retweets['tweet_id'])
            )
            
            
            fig_retweets.update_traces(
                hovertemplate='<b>Retweet Count: %{y}</b><br><a href="%{customdata}" target="_blank">View Tweet</a><extra></extra>'
            )

          
            st.markdown("### Top Tweets with Highest Retweet Count")
            st.plotly_chart(fig_retweets)
        st.divider()
    

        if 'hashtags' in dataframe.columns:
            try:
              
                hashtag_series = dataframe['hashtags'].str.split(',').explode().str.strip().value_counts().reset_index()
                hashtag_series.columns = ['Hashtag', 'Count']
                hashtag_series_complete = hashtag_series.sort_values(by='Count', ascending=False)
                hashtag_series = hashtag_series.sort_values(by='Count', ascending=False)[:20]

                st.markdown("### Hashtag Counts")
                st.markdown(
                    """
                    <style>
                    .wide-table { width: 100%; }
                    .wide-table th, .wide-table td { padding: 8px 20px; text-align: left; }
                    .wide-table th { font-weight: bold; }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.write(hashtag_series.to_html(index=False, classes='wide-table'), unsafe_allow_html=True)

                hashtag_csv = hashtag_series_complete.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Hashtag Count Data as CSV",
                    data=hashtag_csv,
                    file_name="hashtag_count_data.csv",
                    mime="text/csv"
                )

                st.divider()

                hashtag_series_dict = dataframe['hashtags'].str.split(',').explode().str.strip().value_counts().to_dict()
                if hashtag_series_dict:  # Ensure there's data for the word cloud
                    font_path = "JustAnotherHand-Regular.ttf"
                    wordcloud = WordCloud(
                        width=1000, height=500, background_color='black', colormap='tab20b_r',
                        contour_color='black', contour_width=1.5, max_words=200,
                        relative_scaling=0.7, font_path=font_path, max_font_size=280
                    ).generate_from_frequencies(hashtag_series_dict)

                    st.markdown("### Hashtag Word Cloud")
                    plt.figure(figsize=(12, 6))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.write("No valid hashtags found to generate a word cloud.")

            except Exception as e:
                st.error(f"An error occurred while processing hashtags: {e}")

        else:
            st.write("No 'hashtags' column found in the data.")
            st.write("Please ensure the necessary column exists.")

        st.divider()
        st.markdown("### Hashtag-Based User Classification")

        try:
            dataframe['hashtags'] = dataframe['hashtags'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
            dataframe['hashtag_count'] = dataframe['hashtags'].apply(len)

            dataframe['hashtag_classification'] = dataframe['hashtag_count'].apply(
                lambda x: "Bot" if x > 10 else ("Potential Bot" if 5 <= x <= 10 else "Need More Analysis")
            )

            classified_tweets = dataframe[['tweet_id', 'user_screen_name','user_name', 'hashtags', 'hashtag_count', 'hashtag_classification']]
            st.write("### Classified Tweets Based on Hashtags")
            st.write(classified_tweets)

            # CSV Download
            classified_tweets_csv = classified_tweets.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Classified Tweets as CSV",
                data=classified_tweets_csv,
                file_name="classified_tweets.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"An error occurred while classifying tweets: {e}")

        if 'text' in dataframe.columns:
            try:
                import nltk
                from nltk.corpus import stopwords
                from wordcloud import WordCloud
                import matplotlib.pyplot as plt
                import re
                import langdetect
                
                # Download necessary NLTK data
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords')
                
                # Get standard English stopwords
                stop_words = set(stopwords.words('english'))
                
                # Add custom stopwords related to Twitter and common terms
                custom_stopwords = {
                    'rt', 'https', 'co', 'amp', 'http', 'twitter', 'tweet', 'retweet',
                    'user', 'via', 'new', 'today', 'like', 'follow', 'now', 'get',
                    'one', 'will', 'day', 'time', 'see', 'know', 'just', 'people',
                    'going', 'says', 'say', 'said', 'u', 'im', 'dont', 'thats', 'its'
                }
                
                # Combine stop word sets
                all_stopwords = stop_words.union(custom_stopwords)
                
                # Function to detect English language
                def is_english(text):
                    if not isinstance(text, str) or len(text.strip()) < 10:
                        return False
                    try:
                        return langdetect.detect(text) == 'en'
                    except:
                        return False
                
                # Function to clean text
                def clean_text(text):
                    if not isinstance(text, str):
                        return ""
                    
                    # Check if the text is English
                    if not is_english(text):
                        return ""
                        
                    
                    text = text.lower()
                    
                    
                    text = re.sub(r'https?://\S+|www\.\S+', '', text)
                    text = re.sub(r'@\w+', '', text)
                    text = re.sub(r'#', '', text)
                    text = re.sub(r'[^\w\s]', '', text)
                    text = re.sub(r'\d+', '', text)
                    words = text.split()
                    words = [word for word in words if word not in all_stopwords and len(word) > 2]
                    
                    return ' '.join(words)
                
            
                
                cleaned_texts = dataframe['text'].apply(clean_text)
                
               
                all_text = ' '.join(cleaned_texts)
                
                if all_text.strip():  # Check if we have any text after cleaning
                    # Create and display wordcloud
                    st.markdown("### Word Cloud of tweet content")
                    
                    
                    font_path = "JustAnotherHand-Regular.ttf"
                    wordcloud = WordCloud(
                        width=1000, height=500, 
                        background_color='black', 
                        colormap='plasma',  
                        contour_color='black', 
                        contour_width=1.5, 
                        max_words=200,
                        relative_scaling=0.7, 
                        font_path=font_path, 
                        max_font_size=280,
                        random_state=42  
                    ).generate(all_text)
                    
                    plt.figure(figsize=(12, 6))
                    plt.imshow(wordcloud, interpolation="bilinear")
                    plt.axis("off")
                    st.pyplot(plt)
                    
                    
                    word_freq = {}
                    for word in all_text.split():
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] = 1
                    
                else:
                    st.write("No meaningful English text content found after filtering and cleaning.")
                    
            except Exception as e:
                st.error(f"An error occurred while processing text: {e}")
        else:
            st.write("No 'text' column found in the data.")
            st.write("Please ensure the necessary column exists.")

        
        st.divider()

        if 'user_friends_count' in dataframe.columns and 'user_followers_count' in dataframe.columns:
            
            # Convert columns to numeric and handle errors
            dataframe['user_friends_count'] = pd.to_numeric(dataframe['user_friends_count'], errors='coerce').fillna(0)
            dataframe['user_followers_count'] = pd.to_numeric(dataframe['user_followers_count'], errors='coerce').fillna(0)
            
            # Calculate the following-to-follower ratio
            dataframe['following_follower_ratio'] = dataframe.apply(
                lambda row: row['user_friends_count'] / (row['user_followers_count'] if row['user_followers_count'] != 0 else 1),
                axis=1
            )
            
            # Function to classify users based on the ratio and followers
            def classify_user(row):
                if row['user_followers_count'] < 10:
                    # Adjusted conditions for followers < 10
                    if row['following_follower_ratio'] > 100:
                        return 'bot'
                    elif 50 < row['following_follower_ratio'] <= 100:
                        return 'potential bot'
                    else:
                        return 'Need More Analysis'
                else:
                    # Normal conditions for followers >= 10
                    if row['following_follower_ratio'] > 25:
                        return 'bot'
                    elif 15 <= row['following_follower_ratio'] <= 25:
                        return 'potential bot'
                    else:
                        return 'Need More Analysis'
            
            # Apply the classification function
            dataframe['following_follower_label'] = dataframe.apply(classify_user, axis=1)
            
            # Display the data
            st.markdown("### Following-to-Follower Ratio Analysis")
            st.write(dataframe[['user_screen_name','user_name', 'user_friends_count', 'user_followers_count', 'following_follower_ratio', 'following_follower_label']])
            
            # Generate bar chart of classifications
            label_counts = dataframe['following_follower_label'].value_counts().reset_index()
            label_counts.columns = ['Classification', 'Count']
            
            
            
            # Allow download of the data
            ratio_csv = dataframe[['user_name', 'user_friends_count', 'user_followers_count', 'following_follower_ratio', 'following_follower_label']].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Following-to-Follower Ratio Data as CSV",
                data=ratio_csv,
                file_name="following_follower_ratio_data.csv",
                mime="text/csv"
            )
        else:
            st.write("Columns 'user_friends_count' or 'user_followers_count' not found in the data.")

        st.divider()
        if 'user_followers_count' in dataframe.columns:
            dataframe['user_followers_count'] = pd.to_numeric(dataframe['user_followers_count'], errors='coerce').fillna(0)

            bins = [-1, 0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000, float('inf')]
            labels = [
                '0 Followers', 
                '1 Follower', 
                '2-5 Followers', 
                '6-10 Followers',
                '11-50 Followers', 
                '51-100 Followers', 
                '101-200 Followers',
                '201-500 Followers', 
                '500-999 Followers',  
                '1000-2000 Followers',  
                '2000+ Followers'  
            ]

            dataframe['follower_range'] = pd.cut(dataframe['user_followers_count'], bins=bins, labels=labels, right=True)

            follower_distribution = dataframe['follower_range'].value_counts().reindex(labels, fill_value=0)

            st.markdown("### Follower Distribution")
            fig_pie = go.Figure(
                go.Pie(
                    labels=follower_distribution.index.astype(str), 
                    values=follower_distribution.values,
                    hole=0.4  
                )
            )
            fig_pie.update_layout(
                title="Distribution of Accounts Based on Follower Count",
                template='plotly_dark',
                legend=dict(
                    traceorder="normal",  # Preserve the natural order of labels
                    itemsizing="constant",
                    font=dict(size=12),
                    title="Follower Range",
                )
            )
            st.plotly_chart(fig_pie)
        else:
            st.write("Column 'user_followers_count' not found in the data.")

        st.divider()
        if 'user_friends_count' in dataframe.columns:
            dataframe['user_friends_count'] = pd.to_numeric(dataframe['user_friends_count'], errors='coerce').fillna(0)
            
            # Define bins and labels for following count
            bins = [-1, 0, 1, 5, 10, 50, 100, 200, 500, 1000, 2000, float('inf')]
            labels = [
                '0 Following',
                '1 Following',
                '2-5 Following',
                '6-10 Following',
                '11-50 Following',
                '51-100 Following',
                '101-200 Following',
                '201-500 Following',
                '500-999 Following',
                '1000-2000 Following',
                '2000+ Following'
            ]
            
            # Categorize the following counts
            dataframe['following_range'] = pd.cut(dataframe['user_friends_count'], bins=bins, labels=labels, right=True)
            
            # Count occurrences in each range and ensure all labels are present
            following_distribution = dataframe['following_range'].value_counts().reindex(labels, fill_value=0)
            
            # Create pie chart with a different color scheme
            st.markdown("### Following Distribution")
            fig_pie = go.Figure(
                go.Pie(
                    labels=following_distribution.index.astype(str),
                    values=following_distribution.values,
                    hole=0.4,
                    marker=dict(
                        colors=px.colors.sequential.Inferno
, 
                    )
                )
            )
            fig_pie.update_layout(
                title="Distribution of Accounts Based on Following Count",
                template='plotly_dark',
                legend=dict(
                    traceorder="normal",
                    itemsizing="constant",
                    font=dict(size=12),
                    title="Following Range",
                )
            )
            st.plotly_chart(fig_pie)
        else:
            st.write("Column 'user_friends_count' not found in the data.")

        st.divider()
        if 'retweet_count' in dataframe.columns and 'tweet_id' in dataframe.columns:
            top_retweeted_tweets = yt.nlargest(5, 'retweet_count')[['tweet_id', 'retweet_count']]
            st.markdown("### Top 5 Tweets with the Highest Retweet Count")
            for _, row in top_retweeted_tweets.iterrows():
                tweet_id = row['tweet_id']
                st.markdown(f"#### Tweet ID: {tweet_id} (Retweets: {row['retweet_count']})")
                try:
                   
                    api_url = f'https://publish.twitter.com/oembed?url=https://twitter.com/XXX/status/{tweet_id}'
                    response = requests.get(api_url)
                    embed_html = response.json().get("html", "<blockquote class='missing'>This tweet is no longer available.</blockquote>")
                except Exception as e:
                    embed_html = f"<blockquote class='missing'>This tweet is no longer available. Error: {e}</blockquote>"

                st.components.v1.html(embed_html, height=700)
        st.divider()
        if 'user_name' in dataframe.columns and 'tweet_id' in dataframe.columns:
    
            total_tweets = yt['tweet_id'].nunique()

            # Group by user_name to count tweets by each user
            user_tweet_counts = yt.groupby('user_name')['tweet_id'].count().reset_index()
            user_tweet_counts.columns = ['user_name', 'tweet_count']

            
            user_tweet_counts['tweet_percentage'] = (user_tweet_counts['tweet_count'] / total_tweets) * 100

            percent_dt = 2
            high_tweet_users = user_tweet_counts[user_tweet_counts['tweet_percentage'] > percent_dt ]

            
            high_tweet_users['label'] = 'Bot'

          
            st.markdown(f"### Total Tweets in Dataset: {total_tweets}")
            if not high_tweet_users.empty:
                st.markdown(f"### Users Contributing More Than {percent_dt}% of Total Tweets")
                st.write(high_tweet_users)

                
                high_tweet_csv = high_tweet_users.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download High-Contributing Users Data as CSV",
                    data=high_tweet_csv,
                    file_name="high_contributing_users.csv",
                    mime="text/csv"
                )
            else:
                st.write(f"No users have contributed more than {percent_dt}% of total tweets.")

        else:
            st.write("Columns 'user_name' or 'tweet_id' not found in the data.")


        st.divider()
       
       
        # if {'user_screen_name', 'retweet_count', 'reply_count', 'view Count'}.issubset(dataframe.columns):
        #     # Fill NaN values and convert to float
        #     dataframe[['retweet_count', 'reply_count', 'view Count']] = dataframe[['retweet_count', 'reply_count', 'view Count']].fillna(0).astype(float)
            
        #     user_metrics = dataframe.groupby('user_screen_name')[['retweet_count', 'reply_count', 'view Count']].sum().reset_index()
        #     top_users = dataframe.groupby('user_screen_name')[['retweet_count', 'reply_count', 'view Count']].sum().nlargest(5, 'retweet_count').reset_index()
        #     # User selects usernames using multiselect
        #     selected_users = st.multiselect(
        #         "Select Users to Visualize Engagement Metrics",
        #         options=user_metrics['user_screen_name'].tolist(),
        #         default=top_users['user_screen_name'].tolist()[:5],  # Default to top 5 users
        #         help="Select one or more users to visualize their engagement metrics."
        #     )

        #     if selected_users:
        #         # Filter the dataframe for selected users
        #         filtered_metrics = user_metrics[user_metrics['user_screen_name'].isin(selected_users)]
        #         filtered_metrics.set_index('user_screen_name', inplace=True)

        #         # Normalize metrics for radar chart
        #         normalized_metrics = filtered_metrics / filtered_metrics.max()

        #         # Create spider chart
        #         st.markdown("### Engagement Metrics for Selected Users")
        #         fig_engagement_spider = go.Figure()
                
        #         for user in normalized_metrics.index:
        #             fig_engagement_spider.add_trace(go.Scatterpolar(
        #                 r=normalized_metrics.loc[user].values,
        #                 theta=['Retweets', 'Replies', 'Views'],
        #                 fill='toself',
        #                 name=user
        #             ))
                
        #         fig_engagement_spider.update_layout(
        #             polar=dict(radialaxis=dict(visible=True, range=[0, 1]),  
        #                     angularaxis=dict(tickfont=dict(size=10))),
        #             title="Tweet Engagement Metrics",
        #             template='plotly_dark',
        #             showlegend=True
        #         )
        #         st.plotly_chart(fig_engagement_spider)
        #     else:
        #         st.write("No users selected. Please select at least one user.")
        # else:
        #     st.write("Required columns are missing in the dataframe.")
                
        # st.divider()
        if {'user_screen_name', 'tweet_id', 'retweet_count', 'reply_count', 'view Count'}.issubset(dataframe.columns):
            # Fill NaN values and convert to float
            
            yt[['retweet_count', 'reply_count', 'view Count']] = yt[['retweet_count', 'reply_count', 'view Count']].fillna(0).astype(float)
            
            # Calculate total engagement for all tweets
            yt['total_engagement'] = yt['retweet_count'] + yt['reply_count'] + yt['view Count']
            
            # Get top 5 users by total retweet count
            top_users = yt.groupby('user_screen_name')['retweet_count'].sum().nlargest(5).index.tolist()
            
            # Ensure each row in the dataframe represents a unique user-tweet pair
            # Create a unique identifier for each user-tweet pair
            yt['pair_id'] = yt['user_screen_name'] + '_' + yt['tweet_id'].astype(str)
            
            # Create a list of all unique user-tweet pairs
            all_pairs = []
            for _, row in yt.iterrows():
                pair_text = f"{row['user_screen_name']} (Tweet: {row['tweet_id']}) - RT: {int(row['retweet_count'])}, Reply: {int(row['reply_count'])}, Views: {int(row['view Count'])}"
                all_pairs.append(pair_text)
            
            # Get default selections - best tweet for each top user
            default_tweets = []
            for user in top_users:
                # Get the highest engagement tweet for this user
                best_tweet = yt[yt['user_screen_name'] == user].sort_values('retweet_count', ascending=True).iloc[0]
                pair_text = f"{best_tweet['user_screen_name']} (Tweet: {best_tweet['tweet_id']}) - RT: {int(best_tweet['retweet_count'])}, Reply: {int(best_tweet['reply_count'])}, Views: {int(best_tweet['view Count'])}"
                default_tweets.append(pair_text)
            
            # Add debug information
            # st.write(f"Total number of user-tweet pairs: {len(all_pairs)}")
            # st.write(f"Number of unique users: {yt['user_name'].nunique()}")
            # st.write(f"Number of unique tweets: {len(yt)}")
            # st.write(f"Number of unique user-tweet pairs: {yt['pair_id'].nunique()}")
            
            # User selects username-tweet pairs using multiselect
            selected_pairs = st.multiselect(
                "Select Users and Tweets to Visualize Engagement Metrics",
                options=all_pairs,
                default=default_tweets,  # Default to top 5 users' best tweets
                help="Select one or more user-tweet pairs to visualize their engagement metrics."
            )
            
            if selected_pairs:
                # Extract user_name and tweet_id from selection
                selected_users_tweets = []
                
                for pair in selected_pairs:
                    # Split the string to extract username and tweet_id
                    username = pair.split(" (Tweet: ")[0]
                    tweet_id = pair.split(" (Tweet: ")[1].split(")")[0]
                    selected_users_tweets.append((username, tweet_id))
                
                # Filter the dataframe for selected user-tweet pairs
                filtered_data = yt[
                    yt.apply(lambda row: any((row['user_screen_name'] == user and str(row['tweet_id']) == tweet) 
                                                for user, tweet in selected_users_tweets), axis=1)
                ]
                
                # Create a unique identifier for radar chart
                filtered_data['identifier'] = filtered_data['user_screen_name'] + " (Tweet: " + filtered_data['tweet_id'].astype(str) + ")"
                
                # Check if we actually have data for the selected pairs
                if len(filtered_data) > 0:
                    # Extract only the metrics for normalization
                    metrics_df = filtered_data[['identifier', 'retweet_count', 'reply_count', 'view Count']].set_index('identifier')
                    
                    # Calculate maximum values for each metric for normalization
                    max_values = metrics_df.max()
                    
                    # Avoid division by zero
                    for col in max_values.index:
                        if max_values[col] == 0:
                            max_values[col] = 1
                    
                    # Normalize the metrics
                    normalized_metrics = metrics_df / max_values
                    
                    # Create spider chart
                    st.markdown("### Engagement Metrics for Selected Tweets")
                    fig_engagement_spider = go.Figure()
                    
                    for idx in normalized_metrics.index:
                        fig_engagement_spider.add_trace(go.Scatterpolar(
                            r=normalized_metrics.loc[idx].values,
                            theta=['Retweets', 'Replies', 'Views'],
                            fill='toself',
                            name=idx
                        ))
                    
                    fig_engagement_spider.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]  # Normalized range
                            ),
                            angularaxis=dict(tickfont=dict(size=10))
                        ),
                        title="Tweet Engagement ",
                        template='plotly_dark',
                        showlegend=True
                    )
                    st.plotly_chart(fig_engagement_spider)
                else:
                    st.write("No data found for the selected user-tweet pairs.")
            else:
                st.write("No user-tweet pairs selected. Please select at least one pair.")
        else:
            st.write("Required columns are missing in the dataframe.")
        # if {'user_screen_name', 'tweet_id', 'retweet_count', 'reply_count', 'view Count'}.issubset(dataframe.columns):
        #     # Fill NaN values and convert to float
        #     dataframe[['retweet_count', 'reply_count', 'view Count']] = dataframe[['retweet_count', 'reply_count', 'view Count']].fillna(0).astype(float)
            
        #     # Calculate user metrics
        #     user_metrics = dataframe.groupby('user_screen_name')[['retweet_count', 'reply_count', 'view Count']].sum().reset_index()
        #     top_users = dataframe.groupby('user_screen_name')[['retweet_count', 'reply_count', 'view Count']].sum().nlargest(5, 'retweet_count').reset_index()
            
        #     # Find the best tweet for each user (highest engagement)
        #     dataframe['total_engagement'] = dataframe['retweet_count'] + dataframe['reply_count'] + dataframe['view Count']
            
        #     # Get the best tweet for each user
        #     best_tweets = dataframe.sort_values('total_engagement', ascending=False).groupby('user_screen_name').first().reset_index()
        #     best_tweets = best_tweets[['user_screen_name', 'tweet_id', 'total_engagement']]
            
        #     # Create user-tweet pairs for selection
        #     user_tweet_pairs = dataframe[['user_screen_name', 'tweet_id', 'retweet_count', 'reply_count', 'view Count']].copy()
        #     user_tweet_pairs['total_engagement'] = user_tweet_pairs['retweet_count'] + user_tweet_pairs['reply_count'] + user_tweet_pairs['view Count']
            
        #     # Format options as "username (tweet_id) - engagement stats" for display
        #     selection_options = [
        #         f"{row['user_screen_name']} (Tweet: {row['tweet_id']}) - RT: {int(row['retweet_count'])}, Reply: {int(row['reply_count'])}, Views: {int(row['view Count'])}" 
        #         for _, row in user_tweet_pairs.iterrows()
        #     ]
            
        #     # Create default selections based on top users' best tweets
        #     default_tweets = []
        #     for user in top_users['user_screen_name'].tolist()[:5]:
        #         # Get the best tweet for this user
        #         best_tweet_id = best_tweets[best_tweets['user_screen_name'] == user]['tweet_id'].iloc[0]
        #         user_data = user_tweet_pairs[(user_tweet_pairs['user_screen_name'] == user) & 
        #                                     (user_tweet_pairs['tweet_id'] == best_tweet_id)].iloc[0]
                
        #         default_tweets.append(
        #             f"{user} (Tweet: {best_tweet_id}) - RT: {int(user_data['retweet_count'])}, Reply: {int(user_data['reply_count'])}, Views: {int(user_data['view Count'])}"
        #         )
            
        #     # User selects username-tweet pairs using multiselect
        #     selected_pairs = st.multiselect(
        #         "Select Users and Tweets to Visualize Engagement Metrics",
        #         options=selection_options,
        #         default=default_tweets,  # Default to top 5 users' best tweets
        #         help="Select one or more user-tweet pairs to visualize their engagement metrics."
        #     )
            
        #     if selected_pairs:
        #         # Extract user_screen_name and tweet_id from selection
        #         selected_users_tweets = []
                
        #         for pair in selected_pairs:
        #             # Split the string to extract username and tweet_id
        #             username = pair.split(" (Tweet: ")[0]
        #             tweet_id = pair.split(" (Tweet: ")[1].split(")")[0]
        #             selected_users_tweets.append((username, tweet_id))
                
        #         # Filter the dataframe for selected user-tweet pairs
        #         filtered_data = dataframe[
        #             dataframe.apply(lambda row: any((row['user_screen_name'] == user and row['tweet_id'] == tweet) 
        #                                         for user, tweet in selected_users_tweets), axis=1)
        #         ]
                
        #         # Create a unique identifier for radar chart
        #         filtered_data['identifier'] = filtered_data['user_screen_name'] + " (Tweet: " + filtered_data['tweet_id'].astype(str) + ")"
                
        #         # Extract only the metrics for normalization
        #         metrics_df = filtered_data[['identifier', 'retweet_count', 'reply_count', 'view Count']].set_index('identifier')
                
        #         # Calculate maximum values for each metric for normalization
        #         max_values = metrics_df.max()
                
        #         # Avoid division by zero
        #         for col in max_values.index:
        #             if max_values[col] == 0:
        #                 max_values[col] = 1
                
        #         # Normalize the metrics
        #         normalized_metrics = metrics_df / max_values
                
        #         # Create spider chart
        #         st.markdown("### Engagement Metrics for Selected Tweets")
        #         fig_engagement_spider = go.Figure()
                
        #         for idx in normalized_metrics.index:
        #             fig_engagement_spider.add_trace(go.Scatterpolar(
        #                 r=normalized_metrics.loc[idx].values,
        #                 theta=['Retweets', 'Replies', 'Views'],
        #                 fill='toself',
        #                 name=idx
        #             ))
                
        #         fig_engagement_spider.update_layout(
        #             polar=dict(
        #                 radialaxis=dict(
        #                     visible=True,
        #                     range=[0, 1]  # Normalized range
        #                 ),
        #                 angularaxis=dict(tickfont=dict(size=10))
        #             ),
        #             title="Normalized Tweet Engagement Metrics by Tweet",
        #             template='plotly_dark',
        #             showlegend=True
        #         )
        #         st.plotly_chart(fig_engagement_spider)
        #     else:
        #         st.write("No user-tweet pairs selected. Please select at least one pair.")
        # else:
        #     st.write("Required columns are missing in the dataframe.")

        st.divider()
        if 'user_creation_date' in dataframe.columns:
            user_growth = unique_users_df.groupby(unique_users_df['user_creation_date'].dt.date).size().cumsum().reset_index(name="cumulative_users")

            st.markdown("### User Growth Over Time")
            fig_user_growth = go.Figure(go.Scatter(
                x=user_growth['user_creation_date'],
                y=user_growth['cumulative_users'],
                mode='lines',
                line=dict(color='green', width=2)
            ))
            fig_user_growth.update_layout(
                title="User Growth Over Time",
                xaxis_title="Date",
                yaxis_title="Cumulative Users",
                template='plotly_dark'
            )
            st.plotly_chart(fig_user_growth)

        st.divider()
        # st.write("Data in the file:")
        # import pandas as pd

        # # Define the logic to assign the final label
        # def assign_bot_label(row):
        #     # Normalize all labels to lowercase for case-insensitive comparison
        #     labels = [row['hashtag_classification'].lower(), 
        #             row['following_follower_label'].lower(), 
        #             row['bot_label'].lower()]
            
        #     # Check if any of the columns contain 'bot' (case-insensitive)
        #     if 'bot' in labels:
        #         return 'Bot'
            
        #     # Check if one or two columns contain 'potential bot'
        #     potential_bots = labels.count('potential bot')
        #     if potential_bots >= 1:
        #         if potential_bots == 3:
        #             return 'Potential Bot'  # If all three are 'potential bot', label as 'potential bot'
        #         return 'Potential Bot'
            
        #     # Check if all three are labeled 'need more analysis'
        #     if all(label == 'need more analysis' for label in labels):
        #         return 'Need More Analysis'
            
        #     return 'undefined'  # Default case if none of the conditions match

        # # Apply the logic to the dataframe
        # dataframe['Is_bot'] = dataframe.apply(assign_bot_label, axis=1)

        # # Display the dataframe
        # st.write(dataframe[['user_screen_name','user_name','user_description', 'user_friends_count' , 'user_followers_count', 'user_creation_date', 'daily_tweets', 'Is_bot']])

        # # Count the number of each category in 'Is_bot'
        # is_bot_counts = dataframe['Is_bot'].value_counts()

        # # Plot the bar chart using Plotly
        # fig = go.Figure()

        # # Adding bars for each category
        # fig.add_trace(go.Bar(
        #     x=is_bot_counts.index,
        #     y=is_bot_counts.values,
        #     marker=dict(color=['rgba(117, 177, 99, 0.8)', 'rgba(255, 165, 0, 0.8)', 'rgba(216, 31, 31, 0.8)']) ,  # Custom colors for each category
        #     text=is_bot_counts.values,
        #     textposition='auto',  # Display values on top of the bars
        # ))

        # # Customize layout
        # fig.update_layout(
        #     title="Distribution of Accounts by Bot Classification",
        #     xaxis_title="Classification",
        #     yaxis_title="Number of Accounts",
        #     template="plotly_dark",
        #     showlegend=False
        # )

        # # Display the bar chart
        # st.plotly_chart(fig)
        

        if 'created_at' in dataframe.columns:
            dataframe['created_at'] = pd.to_datetime(dataframe['created_at'], errors='coerce')
            dataframe['hour'] = dataframe['created_at'].dt.hour
            dataframe['date'] = dataframe['created_at'].dt.date
    
        # Calculate post frequency by hour of day to detect automation patterns
        if 'hour' in dataframe.columns:
            hourly_posts = dataframe.groupby('hour').size().reset_index(name='count')
            
            # Calculate statistics for identifying abnormal patterns
            total_posts = hourly_posts['count'].sum()
            avg_posts_per_hour = total_posts / 24
            hourly_percentage = (hourly_posts['count'] / total_posts * 100).round(2)
            hourly_posts['percentage'] = hourly_percentage
            
            # Create temporal signature plot (key for detecting coordinated campaigns)
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Post Distribution by Hour ", 
                            "Post Volume Over Time (Campaign Activity Pattern)"),
                vertical_spacing=0.25,
                specs=[[{"type": "bar"}],
                    [{"type": "scatter"}]]
            )
            
            # Add hourly distribution bar chart
            fig.add_trace(
                go.Bar(
                    x=hourly_posts['hour'],
                    y=hourly_posts['percentage'],
                    marker=dict(
                        color=hourly_posts['percentage'],
                        colorscale='Plasma',
                        showscale=False
                    ),
                    name="% of Posts",
                    hovertemplate='Hour: %{x}<br>Posts: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            
            threshold_high = 6 
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 23],
                    y=[threshold_high, threshold_high],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash'),
                    name='Automation Threshold',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[0, 23],
                    y=[4.17, 4.17],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dot'),
                    name='Expected Distribution',
                    hoverinfo='skip'
                ),
                row=1, col=1
            )
            
           
            hourly_posts_sorted = sorted(hourly_posts['count'])
            cum_posts = np.cumsum(hourly_posts_sorted)
            cum_posts = cum_posts / cum_posts[-1]
            gini = 1 - 2 * np.trapz(cum_posts, dx=1/24)
            
           
            if 'date' in dataframe.columns:
                daily_volume = dataframe.groupby('date').size().reset_index(name='count')
                daily_volume['date_str'] = daily_volume['date'].astype(str)
                
                
                daily_volume['rolling_avg'] = daily_volume['count'].rolling(window=3, min_periods=1).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=daily_volume['date_str'],
                        y=daily_volume['count'],
                        mode='lines+markers',
                        name='Daily Posts',
                        line=dict(color='royalblue'),
                        hovertemplate='Date: %{x}<br>Posts: %{y}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=daily_volume['date_str'],
                        y=daily_volume['rolling_avg'],
                        mode='lines',
                        name='3-Day Avg',
                        line=dict(color='darkorange', width=1),
                        hovertemplate='Date: %{x}<br>3-Day Avg: %{y:.1f}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                cv = daily_volume['count'].std() / daily_volume['count'].mean() if daily_volume['count'].mean() > 0 else 0
            else:
                cv = 0  
            
            # Calculate user participation metrics
            # if 'user_screen_name' in dataframe.columns:
            #     # Count posts per user
            #     posts_per_user = dataframe['user_screen_name'].value_counts()
            #     total_users = len(posts_per_user)
            #     avg_posts_per_user = posts_per_user.mean()
            #     median_posts_per_user = posts_per_user.median()
            #     max_posts_per_user = posts_per_user.max()
                
            #     # Calculate concentration metrics
            #     top_10_percent_users = int(total_users * 0.1) if total_users > 10 else 1
            #     posts_by_top_users = posts_per_user.head(top_10_percent_users).sum()
            #     concentration_ratio = posts_by_top_users / total_posts if total_posts > 0 else 0
            # else:
            #     avg_posts_per_user = median_posts_per_user = max_posts_per_user = concentration_ratio = 0
            #     total_users = 0
            
            if 'user_screen_name' in yt.columns:
                total_posts = len(yt)
                user_tweet_counts = yt.groupby('user_screen_name').size().reset_index(name='tweet_count')
                top_users = user_tweet_counts.sort_values(by='tweet_count', ascending=False)
                total_users = user_tweet_counts.shape[0]

                total_tweets = yt.shape[0]
                max_tweets_by_user = user_tweet_counts['tweet_count'].max()

                avg_posts_per_user = total_tweets / total_users if total_users > 0 else 0
                max_posts_per_user = max_tweets_by_user

                top_10_percent_users = max(int(total_users * 0.1), 1)
                posts_by_top_users = top_users['tweet_count'].nlargest(top_10_percent_users).sum()

                concentration_ratio = posts_by_top_users / total_posts if total_posts > 0 else 0
                print(f'Concentration Ratio: {concentration_ratio * 100:.2f}%')
            else:
                avg_posts_per_user = median_posts_per_user = max_posts_per_user = concentration_ratio = 0
                total_users = total_posts = 0

            
            
            # if 'text' in dataframe.columns:
            #     clean_texts = dataframe['text'].astype(str).apply(lambda x: re.sub(r'https\S+', '', x.lower()))
            #     vectorizer = TfidfVectorizer()
            #     tfidf_matrix = vectorizer.fit_transform(clean_texts)
            #     similarity_matrix = cosine_similarity(tfidf_matrix)
            #     avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
            #     duplicate_ratio = avg_similarity
            # else:
            #     duplicate_ratio = 0
            if 'text' in dataframe.columns and 'tweet_id' in dataframe.columns:
                
                clean_texts = yt['text'].astype(str).apply(lambda x: re.sub(r'https\S+', '', x.lower()))
                
               
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform(clean_texts)
                
               
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                
                avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
                duplicate_ratio = avg_similarity
                
                
                duplicate_threshold = 0.8  
                duplicate_pairs = []
                
                
                # for i in range(len(similarity_matrix)):
                #     for j in range(i+1, len(similarity_matrix)):
                        
                #         if yt.iloc[i]['tweet_id'] == yt.iloc[j]['tweet_id']:
                #             continue
                            
                #         if similarity_matrix[i, j] > duplicate_threshold:
                #             duplicate_pairs.append((i, j, similarity_matrix[i, j]))
                
                
                # if duplicate_pairs:
                #     st.markdown("### Potential Duplicate Tweets")
                    
                #     duplicate_df = pd.DataFrame(columns=['Tweet ID 1', 'Original Tweet', 'Tweet ID 2','Similar Tweet', 'Similarity Score'])
                    
                #     for i, j, score in duplicate_pairs:
                #         tweet_id_1 = yt.iloc[i]['tweet_id']
                #         tweet_id_2 = yt.iloc[j]['tweet_id']
                #         original_text = yt.iloc[i]['text']
                #         duplicate_text = yt.iloc[j]['text']
                        
                #         new_row = pd.DataFrame({
                #             'Tweet ID 1': [tweet_id_1],
                            
                #             'Original Tweet': [original_text[:100] + '...' if len(original_text) > 100 else original_text],
                #             'Tweet ID 2': [tweet_id_2],
                #             'Similar Tweet': [duplicate_text[:100] + '...' if len(duplicate_text) > 100 else duplicate_text],
                #             'Similarity Score': [f"{score:.2f}"]
                #         })
                        
                #         duplicate_df = pd.concat([duplicate_df, new_row], ignore_index=True)
                    
                    
                #     duplicate_df = duplicate_df.sort_values('Similarity Score', ascending=False)
                    
                    
                #     st.dataframe(duplicate_df)
                    
                    
                    
                # else:
                #     st.write(f"No duplicate tweets found (using similarity threshold of {duplicate_threshold}).")
                #     st.write(f"Average similarity across all tweets: {avg_similarity:.4f}")
            else:
                duplicate_ratio = 0
                st.write("Text or tweet_id column not found in the dataframe. Cannot analyze tweet duplicates.")
                        
            fig.update_layout(
                height=700,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(title="Hour of Day"),
                xaxis2=dict(title="Date"),
                yaxis=dict(title="Percentage of Total Posts"),
                yaxis2=dict(title="Number of Posts"),
                hovermode="closest",
                template="plotly_white"
            )
            
            
            st.plotly_chart(fig, use_container_width=True)
            
            
            campaign_indicators = {
                "temporal_concentration": {
                    "value": gini,
                    "weight": 0.25,
                    "description": "Measures how concentrated posting is in specific hours (0-1 scale)"
                },
                "volume_volatility": {
                    "value": min(cv, 2) / 2, 
                    "weight": 0.35,
                    "description": "Measures how bursty the posting pattern is"
                },
                "user_concentration": {
                    "value": concentration_ratio,
                    "weight": 0.20,
                    "description": "Portion of content from top 10% of users"
                },
                "content_duplication": {
                    "value": duplicate_ratio,
                    "weight": 0.20,
                    "description": "Portion of messages that are duplicates"
                }
            }
            
            
            campaign_score = sum(indicator["value"] * indicator["weight"] for indicator in campaign_indicators.values())
            campaign_score = min(max(campaign_score, 0), 1)  
            
            
            st.subheader("Campaign Analysis Metrics")
            
           
            col1, col2, col3 = st.columns(3)
            
            with col1:
                
                score_percentage = int(campaign_score * 100)
                st.metric("Campaign Probability Score", f"{score_percentage}%")
                
                
                if score_percentage >= 75:
                    interpretation = "High likelihood of coordinated campaign"
                    color = "red"
                elif score_percentage >= 50:
                    interpretation = "Moderate indications of campaign activity"
                    color = "orange"
                elif score_percentage >= 25:
                    interpretation = "Some campaign-like patterns detected"
                    color = "yellow"
                else:
                    interpretation = "Likely organic conversation"
                    color = "green"
                    
                st.markdown(f"<p style='color:{color};font-weight:bold;'>{interpretation}</p>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Temporal Pattern Score", f"{int(campaign_indicators['temporal_concentration']['value'] * 100)}%")
                st.metric("Content Duplication", f"{int(campaign_indicators['content_duplication']['value'] * 100)}%")
            
            with col3:
                st.metric("User Concentration", f"{int(campaign_indicators['user_concentration']['value'] * 100)}%")
                st.metric("Volume Volatility", f"{int(campaign_indicators['volume_volatility']['value'] * 100)}%")
            
            
            st.subheader("Key Indicators Explained")
            
            metrics_data = {
                "Metric": ["Posts per User", "Hours with Abnormal Activity", "User Concentration Ratio", "Message Duplication"],
                "Value": [
                    f"{avg_posts_per_user:.4f} avg / {max_posts_per_user:.0f} max",
                    f"{len(hourly_posts[hourly_posts['percentage'] > threshold_high])} of 24 hours",
                    f"{concentration_ratio:.1%} (of content from top 10% users)",
                    f"{duplicate_ratio:.2%} of messages"
                ],
                "Interpretation": [
                    "High disparity suggests coordinated amplification" if max_posts_per_user > 10 * avg_posts_per_user else "Normal distribution",
                    "Automated posting likely" if len(hourly_posts[hourly_posts['percentage'] > threshold_high]) > 5 else "Mostly organic pattern",
                    "High concentration suggests campaign" if concentration_ratio > 0.5 else "Distributed conversation",
                    "High duplication indicates messaging coordination" if duplicate_ratio > 0.3 else "Mostly original content"
                ]
            }
            
            st.table(pd.DataFrame(metrics_data))
            
           
            st.subheader("What Makes This Look Like a Campaign?")
            
            characteristics = [
                f"{'✓' if gini > 0.5 else '✗'} Posting concentrated in specific hours",
                f"{'✓' if cv > 1 else '✗'} Bursts of high activity",
                f"{'✓' if concentration_ratio > 0.5 else '✗'} Content dominated by a small group of users ({concentration_ratio:.1%} from top 10%)",
                f"{'✓' if duplicate_ratio > 0.3 else '✗'} High message similarity or duplication ({duplicate_ratio:.2%} of messages)",
                f"{'✓' if len(hourly_posts[hourly_posts['percentage'] > threshold_high]) > 5 else '✗'} Abnormal posting times ({len(hourly_posts[hourly_posts['percentage'] > threshold_high])} hours with high activity)"
            ]
            
            for char in characteristics:
                st.markdown(char)
        
        
        else:
            st.error("Required timestamp columns not found in data")
             
        st.divider()
        st.write("Bot Analysis")
        import pandas as pd

        
        def assign_bot_label(row):
            
            labels = [row['hashtag_classification'].lower(),
                    row['following_follower_label'].lower(),
                    row['bot_label'].lower()]
            
           
            if any('potential bot (media)' in label for label in labels):
                return 'Potential Bot (Media)'
            
           
            if 'bot' in labels:
                return 'Bot'
            
           
            potential_bots = labels.count('potential bot')
            if potential_bots >= 1:
                if potential_bots == 3:
                    return 'Potential Bot'  
                return 'Potential Bot'
            
           
            if all(label == 'need more analysis' for label in labels):
                return 'Need More Analysis'
            
            return 'undefined'  

       
        dataframe['Is_bot'] = dataframe.apply(assign_bot_label, axis=1)

       
        st.write(dataframe[['user_screen_name','user_name','user_description', 'user_friends_count', 'user_followers_count', 'user_creation_date', 'daily_tweets', 'Is_bot']])

        
        is_bot_counts = dataframe['Is_bot'].value_counts()

       
        fig = go.Figure()

        
        colors = {
            'Bot': 'rgba(216, 31, 31, 0.8)',
            'Potential Bot': 'rgba(255, 165, 0, 0.8)',
            'Potential Bot (Media)': 'rgba(75, 192, 192, 0.8)',
            'Need More Analysis': 'rgba(117, 177, 99, 0.8)',
            'undefined': 'rgba(150, 150, 150, 0.8)'
        }

        
        color_list = [colors.get(category, 'rgba(150, 150, 150, 0.8)') for category in is_bot_counts.index]

        fig.add_trace(go.Bar(
            x=is_bot_counts.index,
            y=is_bot_counts.values,
            marker=dict(color=color_list),  
            text=is_bot_counts.values,
            textposition='auto',  
        ))

        
        fig.update_layout(
            title="Distribution of Accounts by Bot Classification",
            xaxis_title="Classification",
            yaxis_title="Number of Accounts",
            template="plotly_dark",
            showlegend=False
        )

        
        st.plotly_chart(fig)

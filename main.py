import time
import json
import schedule
import urllib.robotparser
import re
import heapq
import tracemalloc
import requests
import nltk


from sched import scheduler
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi_users import FastAPIUsers, models


from passlib.context import CryptContext
from parsel import Selector
from collections import Counter
from datetime import datetime
from bson import ObjectId
from flask import Flask
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from typing import Optional


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# create a MongoDB client and connect to the database
app = FastAPI()
client = MongoClient('mongodb://localhost:27017/')
db = client['news_database']
collection = db["news_collection"]
users_collection = db["users"]
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBasic()


# Download the English language model for NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
english_vocab = set(word.lower() for word in nltk.corpus.words.words())

# Initialize lemmatizer and stopwords
lmtzr = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

# create a RobotFileParser object for the website's robots.txt file
rp = urllib.robotparser.RobotFileParser()
rp.set_url("https://www.example.com/robots.txt")
rp.read()


# Filter data paragraph
def clean_paragraph(paragraph):
    # Remove square brackets
    paragraph = re.sub(r'\[.*?\]', '', paragraph)

    # Remove extra spaces
    paragraph = re.sub(r'\s+', ' ', paragraph)

    # Remove special characters and digits
    paragraph = re.sub(r'[^a-zA-Z\s]', '', paragraph)
    paragraph = re.sub(r'\d+', '', paragraph)

    return paragraph.strip()


@app.get("/scrape_news")
async def scrape_news():

    url = "https://news.ycombinator.com/newest"
    response = requests.get(url)
    selector = Selector(text=response.text)
    # Extract data from the HTML using CSS selectors
    titles = selector.css(".title a::text").getall()
    links = selector.css(".title a::attr(href)").getall()
    links = [
        link for link in links if "https" in link and rp.can_fetch("*", link)]
    new_links = []
    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                new_links.append(link)
        except requests.exceptions.RequestException:
            pass
    links = new_links
    points = selector.css(".score::text").re("\d+")
    comments = selector.css("td.subtext a:last-child::text").re("\d+")
    dates = selector.css("span.age a::text").getall()
    ranks = selector.css(".title .rank::text").getall()

    # Extract additional data from each link
    news_data = []
    for title, link, point, comment, date, rank in zip(titles, links, points, comments, dates, ranks):
        # Make a request to the link and extract data using CSS selectors
        article_response = requests.get(link)
        article_selector = Selector(text=article_response.text)
        paragraphs = article_selector.css("p::text").getall()
        images = article_selector.css("img::attr(src)").getall()

        # Filter out non-English paragraphs
        english_paragraphs = []
        for paragraph in paragraphs:
            # Check if the paragraph contains English words
            english_words = [word.lower() for word in nltk.word_tokenize(
                paragraph) if word.lower() in english_vocab]
            if len(nltk.word_tokenize(paragraph)) > 0 and len(english_words) / len(nltk.word_tokenize(paragraph)) > 0.5:
                cleaned_paragraph = clean_paragraph(paragraph)
                if cleaned_paragraph:
                    english_paragraphs.append(cleaned_paragraph)

        # Tokenize, lemmatize, and filter words
        all_words = []
        for paragraph in english_paragraphs:
            # Tokenize the paragraph
            words = word_tokenize(paragraph)
            # Remove stop words and non-alphabetic words
            words = [lmtzr.lemmatize(word.lower()) for word in words if word.lower(
            ) not in stopwords and word.isalpha()]
            all_words.extend(words)

        # Get the most common words and phrases
        counter = Counter(all_words)
        most_common = counter.most_common(10)
        keywords = [word[0] for word in most_common]

        summary_sentences = []
        for paragraph in english_paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_score = 0
                words = nltk.word_tokenize(sentence.lower())
                for keyword in keywords:
                    if keyword in words:
                        sentence_score += 1
                if sentence not in summary_sentences and sentence_score > 0:
                    summary_sentences.append(sentence)

        summary = ' '.join(summary_sentences[:10])

        # Combine the extracted data into a dictionary
        data = {"title": title, "link": link, "points": point, "comments": comment, "dates": date,
                "rank": rank, "paragraphs": summary, "images": images, "keywords": keywords, "comment_list": []}
        news_data.append(data)
    return news_data


@app.get("/add_data_into_json")
async def add_data_into_json():
    """ Fetches the HTML content of the "newest" page 
     on Hacker News and initializes a Scrapy selector to parse it
    """
    url = "https://news.ycombinator.com/newest"
    response = requests.get(url)
    selector = Selector(text=response.text)
    # Extract data from the HTML using CSS selectors
    titles = selector.css(".title a::text").getall()
    links = selector.css(".title a::attr(href)").getall()
    links = [
        link for link in links if "https" in link and rp.can_fetch("*", link)]
    new_links = []
    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                new_links.append(link)
        except requests.exceptions.RequestException:
            pass
    links = new_links
    points = selector.css(".score::text").re("\d+")
    comments = selector.css("td.subtext a:last-child::text").re("\d+")
    dates = selector.css("span.age a::text").getall()
    ranks = selector.css(".title .rank::text").getall()

    # Extract additional data from each link
    news_data = []
    for title, link, point, comment, date, rank in zip(titles, links, points, comments, dates, ranks):
        # Make a request to the link and extract data using CSS selectors
        article_response = requests.get(link)
        article_selector = Selector(text=article_response.text)
        paragraphs = article_selector.css("p::text").getall()
        images = article_selector.css("img::attr(src)").getall()

        # Filter out images that cannot be accessed
        images = [img for img in images if img.startswith(
            ('http://', 'https://')) and requests.get(img).status_code == 200]

        # Sort images in descending order based on size
        try:
            images = sorted(images, key=lambda img: Image.open(
                BytesIO(requests.get(img).content)).size, reverse=True)
        except UnidentifiedImageError as e:
            print(f"Skipped image: {url}. Reason: {str(e)}")

        # Filter out non-English paragraphs
        english_paragraphs = []
        for paragraph in paragraphs:
            # Check if the paragraph contains English words
            english_words = [word.lower() for word in nltk.word_tokenize(
                paragraph) if word.lower() in english_vocab]
            if len(nltk.word_tokenize(paragraph)) > 0 and len(english_words) / len(nltk.word_tokenize(paragraph)) > 0.5:
                cleaned_paragraph = clean_paragraph(paragraph)
                if cleaned_paragraph:
                    english_paragraphs.append(cleaned_paragraph)

        # Tokenize, lemmatize, and filter words
        all_words = []
        for paragraph in english_paragraphs:
            # Tokenize the paragraph
            words = word_tokenize(paragraph)
            # Remove stop words and non-alphabetic words
            words = [lmtzr.lemmatize(word.lower()) for word in words if word.lower(
            ) not in stopwords and word.isalpha()]
            all_words.extend(words)

        # Get the most common words and phrases
        counter = Counter(all_words)
        most_common = counter.most_common(10)
        keywords = [word[0] for word in most_common]

        """

        This code creates a summary 
        by selecting sentences that contain at least one of the specified keywords and 
        appends them to a list of summary sentences.

        """
        summary_sentences = []
        for paragraph in english_paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_score = 0
                words = nltk.word_tokenize(sentence.lower())
                for keyword in keywords:
                    if keyword in words:
                        sentence_score += 1
                if sentence not in summary_sentences and sentence_score > 0:
                    summary_sentences.append(sentence)

        summary = ' '.join(summary_sentences[:10])
        data = {"title": title, "link": link, "points": point, "comments": comment, "dates": date,
                "rank": rank, "paragraphs": summary, "images": images, "keywords": keywords, "comment_list": []}
        news_data.append(data)

    # Write the updated data back to the JSON file
    with open("main.json", "r+") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {"news_data": []}
        data["news_data"] += news_data
        f.seek(0, 0)
        json.dump(data, f, indent=4)
        f.truncate()

    return news_data


@app.get("/add_data_into_mongodb")
async def add_data_into_mongodb():
    """ Fetches the HTML content of the "newest" page 
     on Hacker News and initializes a Scrapy selector to parse it
    """
    url = "https://news.ycombinator.com/newest"
    response = requests.get(url)
    selector = Selector(text=response.text)
    # Extract data from the HTML using CSS selectors
    titles = selector.css(".title a::text").getall()
    links = selector.css(".title a::attr(href)").getall()
    links = [
        link for link in links if "https" in link and rp.can_fetch("*", link)]
    new_links = []
    for link in links:
        try:
            response = requests.get(link)
            if response.status_code == 200:
                new_links.append(link)
        except requests.exceptions.RequestException:
            pass
    links = new_links
    points = selector.css(".score::text").re("\d+")
    comments = selector.css("td.subtext a:last-child::text").re("\d+")
    dates = selector.css("span.age::attr(title)").getall()
    ranks = selector.css(".title .rank::text").getall()

    # Extract additional data from each link
    # news_data = []
    for title, link, point, comment, date, rank in zip(titles, links, points, comments, dates, ranks):
        # Make a request to the link and extract data using CSS selectors
        article_response = requests.get(link)
        article_selector = Selector(text=article_response.text)
        paragraphs = article_selector.css("p::text").getall()
        images = article_selector.css("img::attr(src)").getall()

        # Filter out images that cannot be accessed
        images = [img for img in images if img.startswith(
            ('http://', 'https://')) and requests.get(img).status_code == 200]

        # Sort images in descending order based on size
        try:
            images = sorted(images, key=lambda img: Image.open(
                BytesIO(requests.get(img).content)).size, reverse=True)
        except UnidentifiedImageError as e:
            print(f"Skipped image: {url}. Reason: {str(e)}")

        # Filter out non-English paragraphs
        english_paragraphs = []
        for paragraph in paragraphs:
            # Check if the paragraph contains English words
            english_words = [word.lower() for word in nltk.word_tokenize(
                paragraph) if word.lower() in english_vocab]
            if len(nltk.word_tokenize(paragraph)) > 0 and len(english_words) / len(nltk.word_tokenize(paragraph)) > 0.5:
                cleaned_paragraph = clean_paragraph(paragraph)
                if cleaned_paragraph:
                    english_paragraphs.append(cleaned_paragraph)

        # Tokenize, lemmatize, and filter words
        all_words = []
        for paragraph in english_paragraphs:
            # Tokenize the paragraph
            words = word_tokenize(paragraph)
            # Remove stop words and non-alphabetic words
            words = [lmtzr.lemmatize(word.lower()) for word in words if word.lower(
            ) not in stopwords and word.isalpha()]
            all_words.extend(words)

        # Get the most common words and phrases
        counter = Counter(all_words)
        most_common = counter.most_common(10)
        keywords = [word[0] for word in most_common]

        """

        This code creates a summary 
        by selecting sentences that contain at least one of the specified keywords and 
        appends them to a list of summary sentences.

        """
        summary_sentences = []
        for paragraph in english_paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            for sentence in sentences:
                sentence_score = 0
                words = nltk.word_tokenize(sentence.lower())
                for keyword in keywords:
                    if keyword in words:
                        sentence_score += 1
                if sentence not in summary_sentences and sentence_score > 0:
                    summary_sentences.append(sentence)

        summary = ' '.join(summary_sentences[:10])

        # Combine the extracted data into a dictionary
        data = {"title": title, "link": link, "points": point, "comments": comment, "dates": date,
                "rank": rank, "paragraphs": summary, "images": images, "keywords": keywords}

        # Convert the _id field to ObjectId
        if "_id" in data:
            data["_id"] = ObjectId(data["_id"])

        # Insert the data into the MongoDB collection
        try:
            result = collection.insert_one(data)
        except DuplicateKeyError as e:
            print(f"Duplicate document: {e}")
        print("Inserted document with id: {}".format(result.inserted_id))

    # return news_data

# Update data at mongodb


@app.put("/update_data_in_mongodb/{document_id}")
async def update_data_in_mongodb(document_id: str, title: Optional[str] = None, link: Optional[str] = None,
                                 points: Optional[str] = None, comments: Optional[str] = None,
                                 dates: Optional[str] = None, rank: Optional[str] = None,
                                 paragraphs: Optional[str] = None, images: Optional[str] = None,
                                 keywords: Optional[str] = None):
    # Define the update fields based on the parameters
    update_fields = {}
    if title is not None:
        update_fields["title"] = title
    if link is not None:
        update_fields["link"] = link
    if points is not None:
        update_fields["points"] = points
    if comments is not None:
        update_fields["comments"] = comments
    if dates is not None:
        update_fields["dates"] = dates
    if rank is not None:
        update_fields["rank"] = rank
    if paragraphs is not None:
        update_fields["paragraphs"] = paragraphs
    if images is not None:
        update_fields["images"] = images
    if keywords is not None:
        update_fields["keywords"] = keywords

    # Update the document in the MongoDB collection
    result = collection.update_one(
        {"_id": ObjectId(document_id)}, {"$set": update_fields})

    if result.modified_count == 1:
        return {"message": "Document updated successfully."}
    else:
        return {"message": "Document not found."}


@app.delete("/delete_news_data/{id}")
async def delete_news_data(id: str):
    # Convert the id string to ObjectId
    object_id = ObjectId(id)

    # Delete the document with the specified id
    result = collection.delete_one({"_id": object_id})

    if result.deleted_count == 1:
        return {"message": "Successfully deleted document with id {}".format(id)}
    else:
        return {"message": "Document with id {} not found".format(id)}


@app.delete("/delete_all_data_from_mongodb")
async def delete_all_data_from_mongodb():
    result = collection.delete_many({})
    print("Deleted {} documents".format(result.deleted_count))
    return {"message": "Deleted {} documents".format(result.deleted_count)}


@app.get("/runs")
async def scrape_and_import():
    # Define the functions
    await scrape_news()
    await add_data_into_mongodb()

    # Schedule the functions to run every 30 minutes
    schedule.every(10).minutes.do(scrape_and_import)

    # Enable tracemalloc
    tracemalloc.start()

    await scrape_and_import()

    print('Scraping and importing completed')
    # Run the scheduled tasks indefinitely
    while True:
        schedule.run_pending()
        time.sleep(1)


@app.post("/filter-duplicates")
async def filter_duplicates():
    # Get all the documents in the collection
    documents = collection.find()

    # Create a set to store unique titles
    unique_titles = set()

    # Iterate over each document and remove duplicates
    for document in documents:
        if document['title'] not in unique_titles:
            # If the title is unique, add it to the set and insert the document
            unique_titles.add(document['title'])
            try:
                collection.insert_one(document)
            except DuplicateKeyError as e:
                print(f"Duplicate document: {document['_id']}")
        else:
            # If the title is a duplicate, remove the document from the collection
            collection.delete_one({'_id': ObjectId(document['_id'])})

    # Create an index on the title field
    collection.create_index('title', unique=True)

    # Close the MongoDB client connection
    client.close()

    return {'message': 'Duplicate data filtered successfully.'}

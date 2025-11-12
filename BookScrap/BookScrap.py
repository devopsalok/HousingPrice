import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

book_Data = pd.read_csv('BookScrap/books_dataset.csv')
print("Data",book_Data)

book_Data.isna().sum()
print("Missing Values",book_Data.isna().sum()) 
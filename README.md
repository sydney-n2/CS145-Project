# CS145-Project: WhoIsWho - IND 

### UCLA CS145 Spring 2024 Final Course Project 

### Group Members: Abby Seseri, Benjamin Carrere, Daniel Zuzarte, Megha Velakacharla, Nam Truong, Sydney Ngao 

This repository contains the code to our final project of UCLA's CS145 course from spring quarter of 2024. 

The [WhoIsWho-IND-KDD-2024](https://www.biendata.xyz/competition/ind_kdd_2024/) challenge aims to address the name disambiguation problem that affects scientific literature. The goal is to create a model that is capable of accurately identifying incorrect paper assignments when given an author's profile as input. This project was an attempt at improving the baseline graph convolutional network (GCN) solution for this problem, but was not submitted officially to the competition.  

The main driver of this project is the notebook file titled "IND-WhoIsWho".  The notebook file can be run to reproduce our results. The GCN Updated Model folder contains the code for the graph convolutional network model used in this project, including the baseline provided by the competition and the modified files from our team members. 


### Reproducing Our Results

1. Download our files [here](https://drive.google.com/drive/folders/1f1HjxLVVqhD47UwFa6Y6o9gIjSG26-3G?usp=sharing).
2. Create a vm instance from Google Cloud (we used Deployment Manager).
3. Download the whoiswho-top-solutions folder.
4. Create a bucket on Google cloud and upload the whoiswho-top-solutions folder to the bucket.
5. After you've done that, you can run the code in this Google Colab. 

#!/bin/bash

# https://www.wordfrequency.info/free.asp
while read WORD; do
  	timeout 10 image-scraper -s /random_images https://imgur.com/search?q=$WORD;
	timeout 10 image-scraper -s /random_images https://www.pinterest.ca/search/pins/?q=$WORD;
done < word.list
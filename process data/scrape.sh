export THREADS=5;

COOKIE=''
MIN=$1;
MAX=$2;


if [ -z "$COOKIE" ]; then
    echo "Please navigate to courses.cs50.net, log in via HarvardKey, open the JavaScript console, and evaluate the variable document.cookie and paste it here without quotes. It should look like AWSELB=1234567890; PHPSESSID=1234567890."
    read -p "document.cookie: " COOKIE;
fi;

if [ -z "$MIN" ]; then
    echo "What Q guide ID would you like to scrape?";
    read -p "Course ID: " MIN;
fi;

if [ -z "$MAX" ]; then
    MAX=$MIN;
fi;

if [ -z "$THREADS" ]; then
    read -p "Number of threads: " THREADS;
fi;

mkdir -p data;
for COURSEID in $(seq $MIN $MAX); do
    while [[ `jobs | wc -l` -gt $THREADS ]]; do
        sleep 0.2;
    done;

    printf "\rScraping course $COURSEID..." ;
    curl "https://courses.cs50.net/classes/QForCatNum/$COURSEID" -H "Cookie:$COOKIE" -o data/$COURSEID.json --silent &
done;

echo;

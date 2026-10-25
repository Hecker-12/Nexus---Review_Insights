# Verification Report
Generated at: 2026-10-25 13:13:30 IST

## PART 1: API checks (FastAPI / Uvicorn)
Base URL: `http://127.0.0.1:8000`

### 1.1 GET /sentiment
- **Total reviews**: 50000
- **Positive**: 25331
- **Negative**: 24669
- **% Negative**: 49.34%

First 5 themes in `themes[]`:
| Topic ID | Label | Count | Positive | Negative | % Negative |
|:---:|:---|:---:|:---:|:---:|:---:|
| -1 | Unclustered | 22623 | 11885 | 10738 | 47.46% |
| 0 | Books | 9231 | 4730 | 4501 | 48.76% |
| 1 | Movies and DVDs | 5363 | 2380 | 2983 | 55.62% |
| 2 | Music CDs and albums | 3113 | 1871 | 1242 | 39.9% |
| 3 | Toys and gifts | 1203 | 714 | 489 | 40.65% |

### 1.2 GET /search (Top 3 results per query)

#### Query: `battery died quickly`
- **Rank 1**: Review `1222` | Score: `0.5296` | Theme: `Batteries and chargers`
  - Text: "Roomba Battery: The battery died after 3 wks. I think it may have been defective, but not willing to go through all the "
- **Rank 2**: Review `35556` | Score: `0.5233` | Theme: `Batteries and chargers`
  - Text: "!!! BEWARE OF THIS PRODUCT !!!: No matter how cute this looks. The company carries no spare parts what so ever. So if yo"
- **Rank 3**: Review `1104` | Score: `0.5106` | Theme: `Unclustered`
  - Text: "Stopped working after short period: Stopped working after short period. It sounded good at first, but then became distor"

#### Query: `great jazz album`
- **Rank 1**: Review `11542` | Score: `0.7993` | Theme: `Music CDs and albums`
  - Text: "The Greatest Jazz Album: This is the best version of this album. The remastering job and the extras are excellent."
- **Rank 2**: Review `48413` | Score: `0.7911` | Theme: `Music CDs and albums`
  - Text: "Excellent: This album is one of the best i have listened so far. It is easy and quality listening and wow, excellent. If"
- **Rank 3**: Review `6466` | Score: `0.7263` | Theme: `Music CDs and albums`
  - Text: "Wonderful performance: I love this album. I know many people don't consider this jazz, but I don't pretend to be anythin"

#### Query: `baby bottle leaks`
- **Rank 1**: Review `16647` | Score: `0.7486` | Theme: `Baby bottles and diapers`
  - Text: "Buy some other bottles: I brought these bottles for my baby girl two weeks ago. I wound not recommend anyone buying them"
- **Rank 2**: Review `44139` | Score: `0.7425` | Theme: `Baby bottles and diapers`
  - Text: "The leaky bottle: My daughter was born 4 months ago and have just started to consistanly use bottles. Someone recommende"
- **Rank 3**: Review `24263` | Score: `0.7244` | Theme: `Baby bottles and diapers`
  - Text: "Don't work for everyone: I received a really nice gift pack of Avent bottles at one of my baby showers. The quality of t"

#### Query: `asdfgh`
- **Rank 1**: Review `46438` | Score: `0.3143` | Theme: `Movies and DVDs`
  - Text: "A Cottage on Dartmoor: Stodgy Asquith was such a show-off in his youth! A barber becomes obsessed w/ his manicurist coll"
- **Rank 2**: Review `1998` | Score: `0.3056` | Theme: `Unclustered`
  - Text: "Nasty Astology: Great book to see the true other side of your sign. Yes we always want to see the good but we know we al"
- **Rank 3**: Review `32978` | Score: `0.3033` | Theme: `Unclustered`
  - Text: "Loved it.: My favorite part about this video was its tone -- not too formal/academic, or too devastated-and-desperate-to"

### 1.3 GET /search?q= (Empty query)
- **HTTP Status Code**: `400`
- **Response Body**: `{"detail":"Search query 'q' cannot be empty"}`

### 1.4 GET /summary/-1 and /summary/999
- `GET /summary/-1` status code: `404`
- `GET /summary/999` status code: `404`

---
## PART 2: Dashboard checks (headless, with streamlit.testing.v1.AppTest)

### 2.1 Load dashboard with API reachable
- **Exceptions**: `None`
- **Overview Metrics**: [('Total Reviews', '50,000'), ('Positive', '25,331'), ('Negative', '24,669'), ('% Negative', '49.34%')]
  - Total Reviews: `50,000` (expected 50,000)
  - Positive: `25,331` (expected 25,331)
  - Negative: `24,669` (expected 24,669)
- **'45% unclustered' note text present**: `True`

- **First 5 bars of chart data (label, pct_negative) in order**:
| Rank | Theme Label | % Negative |
|:---:|:---|:---:|
| 1 | Thermometers and humidity gauges | 74.32% |
| 2 | Cables and TV accessories | 69.38% |
| 3 | Baby bottles and diapers | 68.9% |
| 4 | Computer hardware and storage | 68.45% |
| 5 | Magazine subscriptions | 67.35% |

### 2.2 Theme Explorer selection checks (Themes 4, 5, 20, 28, -1)
| Theme ID | Option Label | Mixed theme caution appears? | Exceptions |
|:---:|:---|:---:|:---|
| 4 | `Theme 4: Coffee makers` | **False** | `None` |
| 5 | `Theme 5: Supplements, food and personal care` | **True** | `None` |
| 20 | `Theme 20: Mixed: Gun cases and backpacks` | **True** | `None` |
| 28 | `Theme 28: Fans` | **True** | `None` |
| -1 | `Theme -1: Unclustered` | **False** | `None` |

### 2.3 Simulate API being down (unused port 59999)
- **Traceback present**: `False`
- **Exceptions**: `ElementList()`
- **Message shown**:
```
**Cannot reach the API at `http://127.0.0.1:59999`.**

Make sure the FastAPI server is running:

```
uvicorn src.api.main:app --reload
```
```

### 2.4 Run an empty search
- **Exceptions**: `ElementList()`
- **UI state on empty input**: Suppresses search execution (client-side guard `query and query.strip()`). No errors displayed, idle placeholder preserved.
- **Direct API call on empty search (`fetch_search('')`)**: Returns status code `400` with body `{"detail":"Search query 'q' cannot be empty"}`

---
## PART 3: Data integrity
- **insights.db reviews row count**: `50000`
- **topics_v1_merged.parquet row count**: `50000`
- **review_id equality (`insights.db` vs `topics_v1_merged.parquet`)**: `True`
- **topic_id equality (`insights.db` vs `topics_v1_merged.parquet`)**: `True`
- **len(embeddings.npy)**: `50000`
- **len(embeddings.npy) == reviews row count**: `True`

---
## PART 4: Summary spot-check material (random_state=7)
Themes evaluated: `[12, 19, 23]`. 2 praise or complaint claims sampled per theme.

### Theme 12: Shoes and boots

#### [Complaint Claim]
> "Multiple reviewers reported poor quality and durability issues, with shoes and boots falling apart, peeling, or coming apart at the seams after very little use."

**Cited Review IDs**: `[4188, 5188, 18081]`

- **Review `4188`** (true sentiment: `negative`):
  "Don't Waste Your Money!!: Sure they are soft and warm, but after giving several as Christmas gifts I was mortified how they fell apart after three days of gentle wear. Cheaply made, soles seperate from bootie and a complete waste of money."

- **Review `5188`** (true sentiment: `negative`):
  "Feel nice, don't hold up: I bought a pair in maroon and they were a little short but fine. The problem was that they started coming apart at the seams after 2 months. Will definitely be looking to a different company for a replacement pair."

- **Review `18081`** (true sentiment: `negative`):
  "Meh...: I wore these 2 times & one of the heels started peeling from the rest of it. A lady complimented them but noted that the same thing happened to her. This company was nice enough to refund me my money when I let them know what happened."

#### [Praise Claim]
> "Certain boots and slippers offer great quality, durability, and a proper fit right out of the box."

**Cited Review IDs**: `[15036, 47341]`

- **Review `15036`** (true sentiment: `positive`):
  "Ugg Ascot review: I've purchased Uggs before but I got these as a gift for myself. They fit great and are of much better quality than other slippers that I've purchased."

- **Review `47341`** (true sentiment: `positive`):
  "Excellent boot: This is the boot I've been buying for general purpose for 10 years. An excellent boot - wide enough for my wide feet - fit well right out of the box - leather is tough enough to last many years and out sole is very close to being a 'mountain' out sole. Good support, good shank, good cushion. this my boot of choice of work and light hiking."

### Theme 19: Bedding and pillows

#### [Complaint Claim]
> "Many sheets and blankets suffer from severe shrinking after washing, making them difficult or impossible to fit on beds."

**Cited Review IDs**: `[8048, 21718, 25255]`

- **Review `8048`** (true sentiment: `negative`):
  "Ok, the sheets weren't expensive but WHAT a disappointment!: After one wash the fitted sheet shrank and color leeched onto it so it has grey splotches in places. We don't even have a deep mattress but now the bare mattress shows on the sides. It'll have to go in our next garage sale."

- **Review `21718`** (true sentiment: `negative`):
  "Very Disappointed!!!: I usually have great success with the products I purchase through Amazon. This product was an extreme disappointment. I washed these sheets exactly as recommended and I could barely stretch the fitted sheet across my mattress. If they shrink anymore in future washings, they will not fit at all. Unfortunately, I washed both sets that I purchased before I tried one set on the bed. I will now have to purchase sheets again very soon."

- **Review `25255`** (true sentiment: `negative`):
  "The darn sheets SHRANK: I bought a couple of these sets because they looked so nice. They do look nice, fitted my full size bed really well, with some extra room. I first had my cleaning lady wash the sheets in normal cycle, tumble dry. It shrank. Then I told her to wash the other sheets in delicate cycle and tumble dry low. It still shrank! It shrank so much that I can't fit them on the bed. So dissapointed."

#### [Praise Claim]
> "Certain pillows and water pillows offer great neck support and help users sleep soundly."

**Cited Review IDs**: `[10575, 27226]`

- **Review `10575`** (true sentiment: `positive`):
  "Nana 102A: ABSOLUTELY LOVE THESE PILLOWS! We have not had a neck ache since we started using these pillows. Once the water is adjusted exactly like you like it, you sleep soundly every night."

- **Review `27226`** (true sentiment: `positive`):
  "Love it!: I originally purchased this for my pregnancy, as I was using 5+ pillows for support, and i was tired of chasing them as they fell off the bed all night long. I loved the support and have slept with it ever since. (i even take it with me on trips.) its great because of the refluff-able fill and because you can roll over and not have to move it. I have yet to wash it in the laundermat and see how it turns out.one con, because of its size, it can be difficult to fluff... do it in the morning while you have energy, not at night when you're ready to go to bed."

### Theme 23: Magazine subscriptions

#### [Praise Claim]
> "Many readers find the content thoughtful, inspiring, well-written, and packed with helpful tips and variety."

**Cited Review IDs**: `[2338, 15639, 49392]`

- **Review `2338`** (true sentiment: `positive`):
  "wow: This magazine is full of thoughtful, useful spiritual reflection and advice written by people who are experienced on their own path. It's smart and inspiring, neither overly intellectual nor vapid. Everything is well written, and it's nicely put together, too. A unique magazine."

- **Review `15639`** (true sentiment: `positive`):
  "Perfect for me: I think this magazine is perfect for me. Lots of variety and helpful tips. Wish it was monthly though or at least a bit more thick than it is now for being bi-monthly."

- **Review `49392`** (true sentiment: `positive`):
  "Worth it!: This Artist Mag is informative, inspirational, and well worth ordering. I live somewhat secluded. For me much of my inspiration comes from the encouragement of other artists accomplishments. Information offered concerning art business is extremely helpful. Glad I ordered!"

#### [Complaint Claim]
> "Subscribers complain that magazines have devolved into oversized advertisement catalogs with pushy product placements and thin content."

**Cited Review IDs**: `[8192, 8483, 28003]`

- **Review `8192`** (true sentiment: `negative`):
  "Maxim has gone downhill big time: I've been a loyal subscriber since 1998 and have always enjoyed the magazine. Something happened to them in early 2006 - they've either been bought or have hired a new editor / staff or something. This magazine is absolutely terrible now. All ads, fewer and fewer pictures of babes in each issue, and the hilarious articles are now long gone. My last two issues (Sept and Oct) were in the trash within minutes of pulling them from my mailbox. I'm canceling my subscription."

- **Review `8483`** (true sentiment: `negative`):
  "one big advertisement and full of nothing: Self what? The entire magazine now pushes product on every inch of every page. Their product placements went from "convenient and subtle" to obviously pandering to their revenue sources. Plus the stories and tips are so far out of reality that anyone over 25 with half of a brain couldn't relate. I once enjoyed Self and their monthly self-help plans. Their core message has gotten lost in the sea of product pushing."

- **Review `28003`** (true sentiment: `negative`):
  "Not worth $1 a magazine: I will definitely not be renewing my subscription to Bicycling. The best word that can describe the magazine is "Fluff." It's extremely thin and has ridiculous, stupid content. Besides the usual "eat this after your ride" segment is the "buy this bike" section, which takes up most of the magazine."


---
## PART 5: Error analysis material
### 5.1 15 random reviews where both `pred_tfidf` and `pred_ft` are wrong (random_state=1)

**1. Review ID `16741`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Reviews unhelpful.: So I ordered the product and used it in a variety of ways to try and get superpowers, all as recommended by the reviews here. Unfortunately I didn't generate any superpowers,I'm not even glowing, the only thing it gave me was stage 4 metastatic cancer. Overall 5 stars on the pote"

**2. Review ID `49827`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "Good only for introduction: This book is good only if you want a quick glimpse of what philosophy is. A start, and that is about all you will get from the book. And regarding some of the chapters about Chinese philosophy, you probably would learn better with other books."

**3. Review ID `26436`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "Not up to the standard of his earlier books: Loved his earlier books in the Congo, Amazon etc... In fact I can still remember the best jokes 8 years later. In this one though his style has changed, less off-hand wit and more manic stream of consciousness. There's still plenty of fascinating info on"

**4. Review ID `13077`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "TOUGH TENOR: The product information neglects to tell you of the contributions of little known Jimmy Forrest-like tenor man Claude Bartee, without which the mellow tracks may have turned out to be a somewhat dull affair."

**5. Review ID `28239`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Well, if you are dumb enough to buy it....: If you are dumb enough to buy a book without finding out whats in it, then you deserve the money loss in purchasing the book. If you had read what it says, you would know it doesn't cover D3D. As far as what it does cover, I give it 5 stars. Don't give it"

**6. Review ID `46307`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "ok: This rangefinder seems to work flawlessly. I have no complaints about it. However I will never buy another bushnell product again. They have the worst customer service I have ever seen. I bought a brand new scope off of them to find out that it's defective. It's got a "lifetime warranty" so I se"

**7. Review ID `41297`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "Different perspective: I am a college student and was required to read Lord of the Flies my freshman year of high school. I don't think anyone in my class, even the professor, enjoyed it. Until visiting Amazon.com, I had never met anyone who liked this book. Universiality or something appealing to a"

**8. Review ID `35339`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Great But I Lost It!: I liked it but after attempting to use the little key chain thing it broke and fell off and I lost it."

**9. Review ID `9329`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Mathematical Cellular Neurophysiology: This book is chock full of equations concerning the Hodgkin-Huxley Model, Kinetics of Ionic Channels, Presynaptic & Postsynaptic Transmission, LTP, LTD, quantum vesicular release and reuptake, & a "scant" neural net theory. There are a plethora of 'end of chapt"

**10. Review ID `31939`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "weather is confusing...: ... and will remain so after you read this book. Everything in Buck's book is useful but it is tough to remember all of his rules without a solid grounding in meteorology. The cover's subtitle "a practical book on flying in all kinds of weather" is accurate. This book is abo"

**11. Review ID `11123`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Sold! The only product loyalty i've ever experineced.: Yes, Yes! Horray! The day has finally come. I've found a facial cleanser I never want to give up. Like many i've tried everything that's come out, wasting a lot of money in the process. I gave up and went to a dermatologist after trying to battl"

**12. Review ID `48305`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "Easy to Use: Overall the item works well. If keeping track of one player it is just fine. When trying to play two players it is hard to get from one player to the second player. To play two players it has to go through all four players before getting back to the first player and then to keep strokes"

**13. Review ID `44311`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Yucky Aftertaste: Heard good things about ON products so I decided to try this. It mixes well with soy milk, but it has a weird aftertaste. However, i've tried other products with even worse aftertaste, so it's not a total deal breaker."

**14. Review ID `6076`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "SPECIAL DIRECTORS CUT: Yo yo what up G? Dis movie is sweeeeeeeet! And for reals yo, you get the extra special, dopey dope suger sweet fresh Directors Cut yo! Word yall yaheard me......."

**15. Review ID `18199`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "A-Okay 2: The reputation of Better Homes and Gardens was the reason I chose this as a gift for my nephew; he hasn't reported back as to his success."

### 5.2 8 random reviews where only TF-IDF is right (`pred_tfidf` correct, `pred_ft` wrong) (random_state=1)

**1. Review ID `3003`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "SERIOUS support for "the girls": I felt like my 44D boobs were being herded into industrial-style containers. Wow, it definitely supports them, and they'd better not ask questions!!! I went around all day feeling like my boobs were 'out there', well-managed, and not going anyplace. It's not somethin"

**2. Review ID `34791`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "So far no new releases for the rest of Jem: I love Jem. A lot of people were wondering when the rest were coming out so I went to find Rhino on the web and found the answer there:Are you going to put out any more videos from G. I. Joe or Jem any time soon? We still don't have any current plans for f"

**3. Review ID `9659`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "great pots, lousy lids: I cook all the time, and these pots are great with the exception of the lids. The pots are truly non-stick and easy to clean. However, the lids have a seam between the glass and the metal frame. Whenever you take the lid off a steaming pot, the condensation drips through the"

**4. Review ID `30197`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "I just wish...: I just wish they had released the Memphis DVD 1983 concert separetely from this box set, instead of using it as a bait. I have all their albums, there's no need for another compilation for me. But the Memphis show is another history. It's their best documented live action ever. Well,"

**5. Review ID `12889`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Average unabridged audio: See story summary above. Other than its timely subject, this came off as a somewhat average thriller. Parts of it made you hold on tight and other parts just kept you lingering. Again I must commend Dick Hill's job at the skill and range in reading these types of novels. He"

**6. Review ID `36840`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "Tara: Pleasant reading material, but not in the same league as some of the other Julie Garwood books I have read. A good tale, but not much excitement."

**7. Review ID `17881`**
- True label: `negative`
- FastText predicted: `positive`
- Text (first 300 chars): "PAS DE VERSION STF: MALGRE VOS INDICATIONS LE BLU RAY CONTAGION PARU CHEZ WARNER NE CONTIENT NI VERSION FRANCAISE NI SOUS TITRAGE FRANCAIS. A EVITER POUR LES NON BILINGUES"

**8. Review ID `14961`**
- True label: `positive`
- FastText predicted: `negative`
- Text (first 300 chars): "Book is a book: Have not read book yet. Purchased because writer (Jim Vanecko) was a good friend who recently passed away (January 2011) - this used item was accurately described by seller."


---
## PART 6: Topic quality material

### Merged Theme 0: Books (Total reviews: 9231)

 1. `[Review 14176]`: "Very unprofessional!: Author seems unaccredited and ignorant. She is forcing her opinion on us without any concrete fact"
 2. `[Review 42341]`: "Bad: This book is almost as bad as 3001. The first 25 pages are interesting. The rest is a waste of time, especially the"
 3. `[Review 17912]`: "Southern Rococo, not Gothic: As one who has lived in the south for over 35 years, I found The Cabal to be very disappoin"
 4. `[Review 20582]`: "smugness and venom: Pat Buchanan recommends this book. That should tell you all you need to know about the agenda of the"
 5. `[Review 12313]`: "A must for those who are truth seekers.: This is one of those books that will help truth seekers on thier path. This boo"
 6. `[Review 19196]`: "Very poor and shallow: It's a shame that Sun fellows can only come with so little material about Java performance. Even"
 7. `[Review 37526]`: "This book is wildly good: The gorgeous imagery of the West is what makes this so incredibly amazing. There are lightning"
 8. `[Review 39310]`: "* I WONT BE ABLE TO DO IT JUSTICE *: I am NOT a writer so bear with my lack of talent when it comes to wriiting reviews,"
 9. `[Review 3599]`: "disapointment!: Usually I say when a book it turned into a movie, the movie lacks something. In this case though I felt"
10. `[Review 40712]`: "The Movie Was Better Than The Book: This book was pretty dry, without much color or interior monologue, just leaving me"

### Merged Theme 5: Supplements, food and personal care (Total reviews: 907)

 1. `[Review 35721]`: "Confused after all these raving reviews: It's hard to find a product nowadays that receives such high reviews, especiall"
 2. `[Review 14272]`: "Kava Kava Turvy!: I was telling someone this morning that this brand of Kava Kava was a GREAT supplement for 'restful' s"
 3. `[Review 47310]`: "Warning: Double Rich Chocolate Should Be Recalled: I received a 5lb. container of the Double Rich Chocolate on March 28,"
 4. `[Review 7795]`: "wish I had bought the small can: I really wanted to like this, but it is just not something I seem to be able to adjust"
 5. `[Review 34443]`: "Side Effects: Initially, I had a boost of energy with first time. Later after couple of days , I noticed intestinal pain"
 6. `[Review 16813]`: "low carb, high quality, excellent price: I'm pleased with the overall quality of this product. Double Rich Chocolate tas"
 7. `[Review 14849]`: "A nice taste and easy raw snack: These are really nice to have on hand when you do not feel like dehydrating your own fl"
 8. `[Review 25566]`: "New package; almost the same product; but different directions!: I've used the Golden Cappucino color for a long time an"
 9. `[Review 41074]`: "Very Cheap Smell: I think some of these other reviewers need to get there nose checked or maybe some coffee beans. My pu"
10. `[Review 33927]`: "Buyer Beware of Saftey issues: This product contains Bismuth, which is a by-product of lead and copper. Absorbed into yo"

### Merged Theme 13: Filters, pumps and cleaners (Total reviews: 344)

 1. `[Review 45226]`: "After a month I'm having problems with it.: I purchased this pump one month ago. It worked great for 3 weeks and now wit"
 2. `[Review 49276]`: "JUNK. DO NOT BUY!: The filters fill your room with the smell of a dead animal. It does not go away. I bought 2 sets of f"
 3. `[Review 42817]`: "Lasts no more than 10 days: I am extremely disappointed in performance of this water filter considering previous reviews"
 4. `[Review 16496]`: "Water filter for Kenmore side by side refrigerator: We were happy to find this filter online costing MUCH LESS than thos"
 5. `[Review 42802]`: "bissell sweeper: this is just what i needed for quick pick-ups. great for scatter rugs. works on bare floor. one problem"
 6. `[Review 38892]`: "Bissel Study Sweep Only Goes Straight: The picture made me think the cleaner would turn like the swivel sweeper but it d"
 7. `[Review 19445]`: "biggest piece of garbage ever!!!: first time i used this vacuum the belt broke and some of the brushes on the beater bar"
 8. `[Review 1333]`: "PLEASE SAVE YOUR MONEY: We bought this set of smaller air cleaners along with one of the larger models. Please see my re"
 9. `[Review 21333]`: "Works Fine For Me: We bought this humidifier about 3 months ago because our bedroom was becoming so dry that we were wak"
10. `[Review 33867]`: "Leaking afer one year: I have gone through sooooo many filters and I love this one, so I am getting another. It is still"

### Merged Theme 20: Mixed: Gun cases and backpacks (Total reviews: 159)

 1. `[Review 8449]`: "weak sauce: the shoulder strap is already broken and it has only been 6 weeks. its a weak backpack, considering that i p"
 2. `[Review 11309]`: "A very good, slick and stylish case: I liked the case when I saw it. It is good to hold my A530, the color, material is"
 3. `[Review 3617]`: "P. O. S. not worth the time to look at it: This is a cheap peice of s***. First you can't get a squirt gun in the holste"
 4. `[Review 26023]`: "hard to find item: Sportsmans guide seems to carry items that are hard to find anywhere else. They are good to do busine"
 5. `[Review 22832]`: "Good value but NOT for a 17" laptop: I bought this for my son to use at college. It works great for his books, Walkman a"
 6. `[Review 5056]`: "Highly NOT recommended: Received item and affixed it to the Dickies "holster" I had purchased. Cost $4.99 to buy and $4."
 7. `[Review 11763]`: "Zipper broke at first attempt: Couldnt even use it, zipper broke while putting on the sumo costume. Tried using duct tap"
 8. `[Review 3087]`: "Perfect!: This bag is perfect. I ordered it in "Luscious," and the color is not too bright - it's just right. There is L"
 9. `[Review 11051]`: "When they say chinese they mean it.: The below review says that this sling works for an AK47 IT DOES NOT. Ill find a use"
10. `[Review 28419]`: "Not to bad: I bought this to sight in a new scope. All i really wanted is to get my scope sighted in and this did the tr"

### Merged Theme 28: Fans (Total reviews: 58)

 1. `[Review 37306]`: "It's a Vornado not a Dyson.: For a fraction of the price, you get this wonderful workhorse. No it does not ook as cool a"
 2. `[Review 1753]`: "Be careful: This fan won Maximum PC's Kick A$$ Award which is why I purchased it. Instructions to install are poor at be"
 3. `[Review 29508]`: "won't fan my fire: This was a very poor choice in a car fan for me. It only has one speed that really doesn't blow very"
 4. `[Review 36110]`: "Very disappointing!: We bought this unit for my daughter's room (she's a year old). This unit has a loud hum ALL THE TIM"
 5. `[Review 32823]`: "It's way too loud even on low!: I purchased this after have 3 failed Holmes units, hoping that the 5 star ratings saying"
 6. `[Review 29494]`: "Big and powerful!: It's a nice fan to have when the hot is here. It's big enough, and moves nicely the air around. It ta"
 7. `[Review 30999]`: "not a fan: not a fan of this product. They are hard to put on and it feels like my legs are being suffocated and my circ"
 8. `[Review 20606]`: "fan loudly hums: I was disappointed with this product. I installed 3 on 3 different types of ceiling fans and all three"
 9. `[Review 20941]`: "350z Exhaust: Exhaust is nice, just have some rattling I need to fix underneath the car. The extra power was noticeable"
10. `[Review 38698]`: "It's a 500 dollar fan!: I bought one of these at SEARS also, this thing could not cool a PHONE BOOTH! Had it in my 13'x1"

---
## PART 7: Repo safety
- **`git grep -l -i "AIza"`**: `none`
- **`.env` is in `.gitignore`**: `True`

- **`git status --short`**:
```
M requirements.txt
 M results/notes.md
?? app/
?? results/verification_report.md
```
- **Count of tracked files > 5 MB**: `0`
- **Tracked files > 5 MB**: `None`
- **`data/` ignored**: `True`
- **`models/` ignored**: `True`
- **`.venv311/` ignored**: `True`

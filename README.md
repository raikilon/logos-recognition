# Logos Recognition for Webshop Services
Determine if a website displays some specific logos helps to understand its underlying activity. In particular, the presence of logos such as Visa, PayPal, DHL, etc. shows that a website provides presumably some payment/delivery services typical of webshops. Obtaining such information from a website is useful in several domains, for example: business intelligence, fight against websites that sell illegal or counterfeit products, etc.

This project aims at developing an efficient classifier that determines the presence or absence of logos within a given set of logos. Websites stores logos either as single images or logos are parts of larger images. The classifier should deal with both situations. Obviously, other pieces of information might help to conclude that a website sells online products. In fact the desired classifier should indicate the probability of a website to sell online products based on the presence/absence of given logos and other pieces of information.

**Main objectives:**
* Creation of a classifier to decide whether a website sells online products or not. This classifier is based on the presence/absence of selected logos related to some payment/delivery services (Visa, DHL, etc.). For this version of the classifier, the logos are stored as single images.
* Extend the previous classifier to deal with logos included in larger images (images can contain several different logos).

**Optional objectives**
* Use neural networks to increase the result.


## Structure
This project is subdivided in two main directories namely docs and software.

- [Documentation](/docs)
- [Software](/software)

Please refer to the README inside each directory for additional information.


## Credits
The project was realized by **Noli Manzoni** (nolimanzoni94@gmail.com) for the module [Bachelor Thesis](https://www.ti.bfh.ch/fileadmin/modules/BTI7321-de.xml) at the  [Bern University of Applied Sciences](https://www.bfh.ch).
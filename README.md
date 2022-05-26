# BiometryESEO

## Principe de la technologie utilisée et Fonctionnement  
  
*__Définition d'un réseau de neurones__*  
Un réseau de neuronne est un système informatique s'inspirant du fonctionnement du cerveau humain pour apprendre.  

*__Fonctionnement Général__*  
Il s'agit d'un ensemble de neuronne virtels, disposé en réseaux virtuels.  
Chaque neurone est un point du réseau qui recoit de l'information entrante et emet de l'information sortante.  
Les informations qui circulent sont élementaires et sont des intensités de signaux.  
Certains neurones sont en charge de capter les données extérieurs (les données brutes), il s'agit de la 1ère couche de neurone : Input Layer.  
Les neurones de la 1ère couche vont lire ces données brutes et va s'activer si son morceau de données brute correspond à son activation.  
Cette activation va envoyer des signaux via les connexions (les synapses) aux neurones de la 2ème couche. Les couches intermédiaires se nomment Hidden Layer.  
Les neurones de la 2ème couche vont diviser les signaux en 2 catégories en fonction de la nature des synapses qui peuvent être des synapes (très) activatrices, ou des synapses (très) inhibitrices. Un signal est donc très activateur si ce signal passe par une synapse très activatrice ET si ce signal est d'une forte intensité (= a un grand nombre).  
Le neurone de la 2ème couche va sommer les contributions des signaux activateurs et inhibiteurs et va ensuite ajouter son propre biais. Le neurone s'activera si les signaux activateurs + biais activateur sont plus important que les signaux inhibiteurs + biais inhibiteurs.  
On les appelle les Réseaux de neurone ReLU (Rectifier Linear Unit). Il en existe d'autres...  
De facon général les neurones de toutes les couches intermédiaires (Hidden Layer) collectent des contributions activatrices et inhibitrices et y ajoute un biais, puis calcule le niveau d'activation en fonction de leur résultat et d'une fonction d'activation.  
  
![neurone](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/neurone.jpg)
    
*__Notre utilisation__*  
Nous utiliserons le principe de Deep Learning. Il s'agit d'une forme d'Intelligence Artificielle (= ensemble de techniques dont le but est d'imiter une forme d'intelligence à travers une succession de règles). Ce qui caractérise le Deep Learning c'est le fait qu'on lui donne un but à accomplir et apprendre de lui-même comment l'atteindre. Il faut donc pour cela, utilisé les réseaux de neurones.  
Dans notre cas, il faut fournir un très grand nombre d'exemple d'images de mains différentes (entre 5000 et 6000 images) à un réseau de neurones, suffisement pour qu'il puisse de lui-même apprendre comment s'adapter à notre problème. Au fur et à mesure l'IA saura détecter correctement les images semblables.  
Comme expliqué précedement, nous voulons que notre IA reconnaisse certaines images de mains. Il faut donc donner aux Input Layer des images sous forme de chiffre pour qu'ils puissent les traiter. Pour ce faire, nos images sont en 500x500 pixels et chaque pixel est représenté par une valeur entre 0 et 1 selon l'intensité de couleur/lumière. On a donc un total de 250 000 neurones d'entrées qu'on donnera aux Input Layer.  
Cette information va se propager d'une couche à l'autre. Chaque neurone des couches intermédiaires prenant une valeur numérique, dépendent de toutes les connexions (synapses) entrantes et de leur poids associé, qui défini son niveau d'activation, pour finalement nous renvoyer un résultat avec l'output layer.  
Nous avons plusieurs neurones en sortie, chacun correspondant à des images de mains différentes. Le degré d'activation de chacun des neurones finaux représente le pourcentage de chance que notre image du départ corresponde à une autre image de mains d'après notre réseau.  
Voici un exemple de représentation des neurones output layer :  
![output](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/neurone1.jpg)  
Lors des premières utilisations du réseau de neurone, il y a peu de chance qu'il arrive à trouver la bonne correspondance d'image. Il faut donc lui apprendre son erreur et pour ce faire, on va comparer le résultat qui nous est donné avec celui qu'on attendait de lui. On aura donc un coût : plus le coût est grand, plus notre réseau est éloigné du résultat. On pourra donc savoir quel poids (des connexions) à le plus participer à cette erreur. Il faut donc faire cette opération un très grand nombre de fois aves des images d'entrées différentes : c'est le principe de l'apprentissage.  
Enfin au fur et à mesure, notre IA arrivera à nous donner des résultats correctes. 

*__Les 3 couches que nous utilisons__*  






## Mode d'emploi utilisateur 

*__Utilisation 1 (avec scanner) :__*

1- Poser la main sur le scanner dans le sens indiqué.  
2- Appuyer sur le bouton "Start"  
Pour mettre le snanner en pause appuyer sur le bouton 'pause'  
Pour annuler le processus appuyer sur "annuler"  
![etape1&2](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/notice_mainSurScan_md.jpg)  
3- Attendre le temps du scan  
4- Retirer la main et attendre le résulat :   
Si sur l'ecran s'affiche 'accès autorisé' l'accès est autorisé   
Si sur l'écran s'affiche 'accès refusé' l'accès est refusé   

![This is an image](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/notice_autorise_md.jpg)
![This is an image](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/notice_refuse_md.jpg)  


*__Utilisation 2 (sans scanner, manuellement) :__*

1- Prendre en photo la main de la personne a qui il faut vérifier l'accès  
2- Mettre la photo prise dans le dossier "traitement"   
3- Attendre que l'IA compare avec le dataset des mains   
4- Attendre résultat :   
Si sur l'ecran s'affiche 'accès autorisé' l'accès est autorisé  
Si sur l'écran s'affiche 'accès refusé' l'accès est refusé  

*__Initialisation :__*

1- La première étape consiste en la prise d'une quinzaine de photo d'une main a qui on veut autorisé l'accès.  
2- Renouveler l'opération autant de fois que nécessaire pour toutes les personnes à qui on autorise l'accès.  
3- Une fois les images prises, les stockées dans le dossier "dataset_main".  
4- Faire tourner l'IA, pour qu'elle puisse se réentrainer.  
  
  ![etape3](https://github.com/parutech/BiometryESEO/blob/main/biblioth%C3%A8que_image/notice_initialisation_md.jpg)


## Guides
- [Utilisation de Git](https://www.atlassian.com/fr/git/tutorials/comparing-workflows/gitflow-workflow)
- [Syntaxe Markdown](https://www.markdownguide.org/basic-syntax/)

## Bibliographie
Paul
- [Biometric Authentication: A Review (2009)](https://www.biometrie-online.net/images/stories/dossiers/generalites/International-Journal-of-u-and-e-Service-Science-and-Technology.pdf)
- [Evaluation of Electrocardiogram for Biometric Authentication (2011)](https://www.scirp.org/pdf/JIS20120100004_57389606.pdf)
- [Biometric Authentication: System Security and User Privacy (2012)](http://biometrics.cse.msu.edu/Publications/SecureBiometrics/JainNandakumar_BiometricAuthenticationSystemSecurityUserPrivacy_IEEEComputer2012.pdf)
- [A Machine Learning Framework for Biometric Authentication Using Electrocardiogram (2019)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8756039)

Laure
- [Reconnaissance biométrique sans contact de la main intégrant des informations de forme et de texture](https://hal.archives-ouvertes.fr/hal-00091740/document)

Charles 
- [Biométrie et normes](https://www.itu.int/net/itunews/issues/2010/01/pdf/201001_05-fr.pdf)
- [Technique de controle d'accès par biométrie](https://clusif.fr/wp-content/uploads/2015/10/controlesaccesbiometrie.pdf)
- [La biométrie et L'IoT](https://www.journaldunet.com/ebusiness/internet-mobile/1508189-comment-la-biometrie-va-t-elle-changer-la-technologie-iot-et-les-pratiques-commerciales/)
- [Thalès : la biométrie au service de l'identification](https://www.thalesgroup.com/fr/europe/france/dis/gouvernement/inspiration/biometrie)
- [Biométrie par la main](https://www.abiova.com/biometrie)
- [Biométrie par signaux psychologiques](https://tel.archives-ouvertes.fr/tel-00778089/document)

Morjana
- [Identification des personnes par fusion de différentes modalités biométriques (2014)](https://hal.archives-ouvertes.fr/tel-01206294/document)

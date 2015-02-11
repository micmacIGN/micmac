1) Verifier et interfacer ImEtProf2Terrain

2) - Inverser les bornes de Z sur certaines geometrie
   - Regler le pas a partir d'une distance inter-PDV,
     donnee soit de maniere explicite, soit en appelant
     une methode virtuelle Centre de PDV

3)  Implanter la classe, avec option 1 ou 2 dimension 
de //




/************************************/
/* convention de codage de  MICMAC. */
/************************************/

Sauf oubli, j'ai respecte dans le code
(y compris le code genere automatiquement)
les conventions de nommage suivantes :

   - les noms de classes commencent tjs par c
   - les noms de membre de classes commencent tjs par m
   - les noms de membre de parametre commencent souvent par  a

On trouvera souvent qqch comme :

cAClasse::cAClasse(const cValue & aValue)  :
   mValue (aValue)
{
}

    - les noms d'enum (type comme champs enumere) commencent par e

/************************************/
/*  Code Genere                     */
/************************************/

Le fichier de parametre de MICMAC est un fichier xml.
La specification formelle de ce fichier est donne
par le fichier xml :
   
          applis/MICMAC/ParamMICMAC.xml

Ce fichier  a en fait deux objets :
 
    - il est utilise lors du lancement de l'executable
    pour faire la plupart des verification sur le fichier 
    de parametre xml (en fait essentiellement verifer que les arbre sont
    appariable)

    - il est utilise pour genere automatiquement le code
    des fichier applis/MICMAC/cParamMICMAC.h et
    applis/MICMAC/cParamMICMAC.cpp;  il definissent
    une classe C++ qui representera completement le
    parametrage de la classe; ce code contient :

          - la definition de la classe
          - des accesseur sur tous les membres
          - les fonction assurant l'initialisation
          a partir du fichier de parametre xml (c'est
          l'interet essentiel)
          - les fonctions permettant de generer du xml
          (interet essentiellement pour la mise au point);


    - on peut bien sur consulter ces 2 fichies generes,
    j'ai essaye de generer un code a peu pres lisible;
    par contre il est dangereux, et completement inutile,
    de les modifier car il seront ecrases sans vergogne
    a la prochaine generation de code;

/************************************/
/* Commentaires                     */
/************************************/

  Il sont repartis entre le ParamMICMAC.xml,
les ".cpp" et les ".h". Il est donc conseilles
de jeter d'aboird un coup d'oeil a l'ensemble.

/************************************/
/* Organisation des fichier MICMAC. */
/************************************/

Tous les fichiers sont sur ELISE/appli/MICMAC
a l'exception du Makefile MakeMICMAC qui est sur ELISE.

Il y a en general un fichier source par classe definie
dans MICMAC.h a l'exception  :

1- des petites classes appartenant
a un meme "module" logique qui peuvent etres
regroupees sous un meme nom de fichier :

     cGeomXXX.cpp : qui regroupe les services de correspondance
     entre la geometrie terrain discrete et la geometrie terrain
     reelle (classe  cGeomDiscR2 et cGeomDiscFPx)

     cEtapeMecComp.cpp : qui regroupe les services permettant
     d'acceder a une etape de mise en correspondances

2- De la classe maitresse cAppliMICMAC dont l'implementation
est eclatee en plusieurs fichiers :

   cAppliMICMAC.cpp  : qui contient toute les fonction
   permettant de creer une appli avec  lecture complete 
   des parametres, ainsi que les accesseurs.

   cAppliMICMAC_MEC.cpp  : qui contien le "squellette" du
   programme de mise en correspondance
     

   cAppliMICMAC_Export.cpp  : qui contiendra les programmes d'exports



Les source correspondant a des binaires sont :

bin/GenParamMICMAC : pour generer  cParamMICMAC.cpp 
et cParamMICMAC.h a partir de ParamMICMAC.xml


TestMICMAC.cpp et bin/TestParamMICMAC : deux prog de test.

SaisieLiaisons.cpp : un programme de saisie de point de 
liaison que j'ai repris de l'ancien correlateur en faisant
le minimum d'adaptation. Il n'a pas ete mise au propre
et ne fait pas vraiment partie du correlateur.


/****************************************/
/*          HISTO-PIEGES                */
/****************************************/

----------1----------

Sur mission Amiens/Penard

ValSpecNotImages  a 0 (par copier colle d'une mission
de Marseilles), image tres sombres, generes a basse
resolution plein de 0 . D'ou correl indeterminees





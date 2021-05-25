# Prérequis

Micmac nécessite pour fonctionner la présence des outils suivants sur votre système :
- [make](http://www.gnu.org/software/make) pour la gestion de taches en parallèle,
- *convert*, d'[ImageMagick](http://www.imagemagick.org), pour la conversion de format des images,
- [exiftool](http://www.sno.phy.queensu.ca/~phil/exiftool) et [exiv2](http://www.exiv2.org) pour la lecture/écriture des métadonnées image,
- [proj4](http://trac.osgeo.org/proj/) pour la conversion de systèmes de coordonnées.

Ces outils s'installent facilement à l'aide de la commande suivante sur les ditributions Debian/Ubuntu :
`sudo apt-get install make imagemagick libimage-exiftool-perl exiv2 proj-bin qt5-default`

Vous pouvez vérifier que Micmac trouve correctement les programmes ci-dessus grâce à la commande :
`bin/mm3d CheckDependencies` (si vous vous trouvez dans le répertoire de Micmac)

La mention *NOT FOUND* indique que l’exécutable correspondant n'est pas présent ou qu'il n'a pu être trouvé. Dans ce cas, vérifiez que les chemins d’accès sont bien inclus dans la variable d'environnement *PATH*.
Il convient de noter que quelle que soit la valeur de la variable *PATH*, le répertoire *binaire-aux* est examiné lors de la recherche des outils externes, mais qu'un outil trouvé ailleurs aura la prépondérance.

## Sous Windows

Les binaires précompilés sont créés avec Visual C++ 2010, il faut donc installer les fichiers nécessaires a leur exécution : [Visual C++ 2010 runtime redistribuables](http://www.microsoft.com/fr-fr/download/details.aspx?id=5555)
Que vous utilisiez la version précompilée ou que vous recompiliez vous-même, il est également requis d'installer :
- [Visual C++ 2005 runtime redistribuables](http://www.microsoft.com/fr-fr/download/details.aspx?id=3387),
- et [Net Framework 2.0](http://www.microsoft.com/fr-fr/download/details.aspx?id=1639).

Il est nécessaire que l'une des variables d'environnement *WINDIR* ou *SystemRoot* soit renseignée avec le nom complet du
répertoire d'installation de Windows (généralement `C:\Windows`), ceci afin d’éviter la confusion entre `convert.exe` d'*ImageMagick* et celui de Windows.
Par mesure de commodité, l'archive des sources, ainsi que l'archive des binaires précompilés pour Windows, contient une version des utilitaires prérequis. En effet, cet OS ne dispose pas d'un service d'installation et de mise à jour de paquets, il peut donc être fastidieux de récupérer les exécutables nécessaires.

# Compilation a partir de l'archive des sources

## Prérequis

En plus des prérequis précédent, la compilation des sources nécessite l'installation de [cmake](www.cmake.org) ainsi que des fichiers d’en-tête de X11 pour Linux et MacOS X.
Le nom du paquet Linux des headers X11 est généralement `libx11-dev`.
Sous Windows, les outils graphiques de saisie ne sont pas générés.
Les utilisateurs de Windows auront besoin de la bibliothèque Qt5 pour générer les interfaces graphiques telles que *SaisieMasqQT*.

Pour optimiser la recompilation, [ccache](ccache.dev) est automatiquement utilisé si détecté.

## Sous Linux / MacOS X

- fait un clone depuis le git repo: `git clone https://github.com/micmacIGN/micmac.git`
- entrez dans le répertoire 'micmac' : `cd micmac`,
- créez un répertoire pour accueillir les fichiers intermédiaires générés par la compilation et placez vous a l’intérieur : `mkdir build & cd build`,
- lancez la generation des makefiles : `cmake ../`
- lancez la compilation en indiquant le nombre de cœurs à utiliser : `make install -j*nombre de cœurs*` (par exemple : `make install -j4).

## Sous Windows (avec Visual C++)

Les premières étapes sont semblables à la compilation sous Linux mais la procédure varie après l'appel a `cmake`.
*Cmake* génère une solution `micmac.sln`, ouvrez-la avec Visual C++ et générez le projet `INSTALL`.
Attention à passer la configuration sur *Release*, les exécutables en mode *Debug* sont bien plus lents a l’exécution.
Si vous construisez la solution complète et non le projet `INSTALL`, les fichiers ne seront pas copiés dans le répertoire `bin`.

# Tester l'installation

Sur le site [logiciels.ign.fr](http://logiciels.ign.fr/?Telechargement,20) se trouve également disponible en téléchargement le jeu de test `Boudha_dataset.zip`.
Le script et les données qu'il contient, permettent de tester si votre installation de Micmac fonctionne correctement. Pour lancer le script, si vous vous trouvez dans le répertoire *Boudha* issu du fichier zip, entrez une commande de la forme :
    
`./boudha_test.sh mon_micmac_bin/`
  
où 'mon_micmac_bin' est le chemin (absolu ou relatif) du répertoire 'bin' de votre installation. Par exemple : `./boudha_test.sh ../micmac/bin/`
Attention, le nom du répertoire doit impérativement finir par un '/' terminal (ou un '\' sous Windows, les deux formes sont possibles).
Lorsque le calcul sera terminé, vous pourrez vérifier le résultat grâce aux trois fichier 'ply' contenus dans le répertoire 'MEC-6-Im'.
Il s'agit du même exemple Boudha que dans la documentation. Éditez les fichiers PLY à l'aide d'un logiciel comme Meshlab pour vérifier que les résultats sont cohérents.

# Notes diverses

Si vous souhaitez utiliser les commandes de Micmac à partir de n'importe quel répertoire sans avoir a spécifier son chemin, il faut ajouter le chemin du répertoire `bin` dans la variable d'environnement `PATH`. Il n'est pas nécessaire d'y ajouter le chemin du répertoire `binaire-aux`.

Sous Linux / MacOS, pour utiliser les outils Qt, ajoutez le chemin du répertoire `lib` dans `LD_LIBRARY_PATH`, dans le fichier `.bashrc` comme suit : `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/micmac/lib/`

Sous MacOSX, si vous voulez utiliser les outils QT avec les binaires précompilés disponibles sur [logiciels.ign.fr](http://logiciels.ign.fr/?Telechargement,20), vous devez installer les librairies Qt pour Mac depuis [http://download.qt-project.org](http://download.qt-project.org/archive/qt/4.8/4.8.4/qt-opensource-mac-4.8.4.dmg).

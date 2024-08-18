# Joe

###

Christophe Meynard 

Christophe.Meynard@ign.fr 

IGN/ENSG/CC-IMI   -  LASTIG/ACTE\
\

Chaine de compilation

Portage Windows

Développements bas niveau


# Compilation, Github

###

Plateformes:

  - **Linux(g++, clang++)**
  - Windows (msvc++)
  - _MacOs (clang++)_


\
\
Compilation (cf. Readme):

  - Compilation/installation mm3d (micmac v1)
  - mkdir build; cd build; cmake ..
  - make full (make ; MMVII GenCodeSymDer ; make)
  - make
\
\
  - ccache : accélère recompilation

###

GitHub (master): 

  - Linux/clang++,  Windows/msvc++, Bench 1
  - Binaire pour windows:
\tiny https://github.com/micmacIGN/micmac/releases/download/Windows_MMVII_build/mmvii_windows.zip \normalsize
  - Manuel MMVII: 
\tiny https://github.com/micmacIGN/micmac/releases/download/MMVII_Documentation/Doc2007_a4.pdf \normalsize

\
\
Binding python :
  
  - En cours, prometteur
  - Mapping de classes MMVII en python\
  (1 par 1, à la "main")
  - Linux uniquement pour l'instant (MacOs?)
  - Wheel => orienté utilisateur final

# Ligne de commande

###
**MMVII** _Command_ _Obl1_ _Obl2_... _[Opt1=xxx]_ _[Opt2=xxx]_...

\
MMVII help

MMVII Command help [Help] [HELP]

\

Attention aux caractères spéciaux dans les paramètres

  - Bash: * ? # ~ ; & " ' ` ! $  \\ | ( ) [ ] { }  < >\
=> Escape par **"**, **'** ou **\\**

  - Cmd.exe: * ? & | ( ) < > ^ "\
=> Escape par **"** ou **^**


# Completion automatique (TAB)

###

- bash (windows aussi)\
Necessite python (python3) \
Cf Readme.md\
\

- Basé sur **MMVII GenArgsSpec Quiet=1**\
\

- => automatique pour nouvelles commandes\
\

- => pertinence depend definition commande\
\

- MMVII doit être dans le PATH et compilé 

# vCommand

###

**vMMVII**
\
\

- GUI pour aide saisie paramêtres commandes

- Necessite Qt. Compilé automatiquement si Qt trouvé

- Cf Readme.md

- Historique des chantiers

- Historique des commandes par chantier

- Edition commandes de l'historique

- Basé sur **MMVII GenArgsSpec Quiet=1**


# Questions ?

###

Récupéré la dernière version ?
\
\
Lu et installé dépendances pour api python ?

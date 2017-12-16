#ifndef CREO_APPLI_H
#define CREO_APPLI_H


class cREO_Appli
{
public:
    cREO_Appli();

    // on commence par déterminer une liste de paires d'images et on sauve cette liste dans un xml
    void computeImPairFromOrthoOverlap();
    // charge la liste de paire
    void loadImPair();

   private:
     // liste de couple d'images
     cSauvegardeNamedRel aSNR ;
     std::string mFileClpIm;
     //= StdGetFromPCP(mFileHom,SauvegardeNamedRel);

     std::string mPatOrt;
     std::string mPatPrio; // les images qu'on ne souhaite pas égaliser.
};

#endif // CREO_APPLI_H

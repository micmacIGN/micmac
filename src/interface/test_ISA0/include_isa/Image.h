#ifndef DEF_IMAGE
#define DEF_IMAGE

#include "PaireImg.h"

class Image : public std::vector<PaireImg> 
{
public:
	Image();
	Image(std::string nom, std::string dossier);

  void SetNum(int n);
  void SetNomImg(std::string l);
  void SetChemin(std::string l);
  void SetNbPts(int n);
  void SetAdressePt(int n);

  int GetNum() const;
  std::string GetNomImg() const;
  std::string GetChemin() const;
  int GetNbPts() const;
  int GetAdressePt() const;
  int GetNbPaires() const;

  PaireImg* AddPaire(PaireImg p);

  bool operator == (const Image& img)const;
  bool operator < (const Image& img)const;
  bool operator == (const std::string nom)const;

private:
	int num;
	static int nbImg;
	std::string nomImg;
	std::string chemin;
	int nbPts;
	int adressePt;//pointeur vers le premier point de l'image dans listInit (optimisation de la recherche d'homologues)
	int nbPaires;
};


typedef std::vector<Image> ListImg;

#endif

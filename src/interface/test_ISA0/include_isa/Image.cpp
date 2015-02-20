#include "StdAfx.h"
#include "Image.h"


using namespace std;


  Image::Image() : num(-1), nomImg("") {}
  Image::Image(string nom, string dossier): nomImg(nom), chemin(dossier) {num=++nbImg; nbPaires=0;}

  void Image::SetNum(int n) {num=n;}
  void Image::SetNomImg(string l) {nomImg = l;num=++nbImg;}
  void Image::SetChemin(string l) {chemin = l;}
  void Image::SetNbPts(int n) {nbPts = n;}
  void Image::SetAdressePt(int n) {adressePt = n;}

  int Image::GetNum() const {return num;}
  string Image::GetNomImg() const {return nomImg;}
  string Image::GetChemin() const {return chemin;}
  int Image::GetNbPts() const {return nbPts;}
  int Image::GetAdressePt() const {return adressePt;}
  int Image::GetNbPaires() const {return nbPaires;}

  PaireImg* Image::AddPaire(PaireImg p) {
	push_back(p);
	nbPaires++;
	return &(at(nbPaires-1));
  }

  bool Image::operator == (const Image& img)const {
	 // if (num==img.GetNum()) {return true;}
	  if (nomImg==img.GetNomImg()) {return true;}
	  else {return false;}
  }
  bool Image::operator < (const Image& img)const {
	  if (num<img.GetNum()) {return true;}
	  else {return false;}
  }
  bool Image::operator == (const string nom)const {
	  if (nom==nomImg) {return true;}
	  else {return false;}
  }

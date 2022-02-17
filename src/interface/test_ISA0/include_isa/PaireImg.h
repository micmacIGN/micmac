#ifndef DEF_PAIREIMG
#define DEF_PAIREIMG

#include "addon_ParamChantierPhotogram.h"

class PaireImg{
public:
  PaireImg();
  PaireImg(int n, ElSTDNS string nom);

  void SetNbPtsInit(int n);
  void IncrNbPts();
  void SetIsAlign(bool b);

  int GetNumImg2() const;
  ElSTDNS string GetNomFichier() const;
  int GetNbPtsInit() const;
  int GetNbPts() const;
  bool GetIsAlign() const;
 
  bool operator < (const PaireImg& paire)const ;
  bool operator == (const PaireImg& paire)const ;
  bool operator == (const int num)const ;

  //Recouvrement
 /* float GetLimitRecouvrement(int i);
  Pt2dr* GetRecouvrement();
  void SetRecouvrement(int i, Pt2dr pt);*/

  //Alignement :
  float GetA() const;
  float GetB() const;

  float DistAddPt(float x, float y)const;
  float DistRemovePt(float x, float y)const;
  void RecalculeAddPt(float x, float y, float dmax);
  void RecalculeRemovePt(float x, float y);

private:
	int numImg2;
	ElSTDNS string nomFichier;
	int nbPtsInit;
	int nbPts;

	  //Recouvrement
	 // Pt2dr recouvrement[4];//xmin, ymin, xmax, ymax

	//Alignement :
	bool isAlign;
	float a;
	float b;
	float moment[5];
};

#endif

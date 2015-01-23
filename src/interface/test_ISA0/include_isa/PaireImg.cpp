#include "StdAfx.h"
#include "PaireImg.h"


using namespace std;


  PaireImg::PaireImg() : numImg2(-1) {}
  PaireImg::PaireImg(int n, ElSTDNS string nom): numImg2(n), nomFichier(nom) {
	nbPtsInit=0;
	nbPts=0;

	isAlign=true;
	for (int i=0; i<5; i++)
		moment[i]=0;
  }
  
  void PaireImg::SetNbPtsInit(int n) {nbPtsInit=n;}
  void PaireImg::IncrNbPts() {nbPts++;}
  void PaireImg::SetIsAlign(bool b) {isAlign=b;}

  int PaireImg::GetNumImg2() const {return numImg2;}
  ElSTDNS string PaireImg::GetNomFichier() const {return nomFichier;}
  int PaireImg::GetNbPtsInit() const {return nbPtsInit;}
  int PaireImg::GetNbPts() const {return nbPts;}
  bool PaireImg::GetIsAlign() const {return isAlign;}
 
  bool PaireImg::operator < (const PaireImg& paire)const {
	  if (numImg2<paire.GetNumImg2()) {return true;}
	  else {return false;}
  }
  bool PaireImg::operator == (const PaireImg& paire)const {
	  return (numImg2==paire.GetNumImg2()) ;
  }
  bool PaireImg::operator == (const int num)const {
	  return (numImg2==num) ;
  }

//Alignement
  float PaireImg::DistAddPt(float x, float y)const{
	float ps=x+(y-b)*a;
	float d=1+a*a;
	float x1=x-ps/d;
	float y1=y-b-ps*a/d;
	return x1*x1+y1*y1;
  }

  float PaireImg::DistRemovePt(float x, float y)const{return 0;}

  void PaireImg::RecalculeAddPt(float x, float y, float dmax){
	nbPts++;
	if (nbPts>2 && isAlign) isAlign=(DistAddPt(x,y)<dmax*dmax);

	if(isAlign) {
		moment[0]+=x;
		moment[1]+=y;
		moment[2]+=x*x;
		moment[3]+=x*y;
		moment[4]+=y*y;
		if (nbPts>1){ 
			float A=(nbPts)*moment[3]+(-moment[1])*moment[1];
			float B=(-moment[1])*moment[3]+(moment[2])*moment[1];
			float D=moment[2]*nbPts+moment[0]*moment[0];
			a=A/D;
			b=B/D;
		}
	}
  }

  void PaireImg::RecalculeRemovePt(float x, float y){}


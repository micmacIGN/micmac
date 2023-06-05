#ifndef _PSEUDOXML_
#define _PSEUDOXML_

#include "Stringpp.hpp"
#include <fstream>
#include <vector>

class PseudoXML{
 public:
  PseudoXML(bool maj=true){_maj=maj;}
  PseudoXML(const stringpp & adresse,bool maj=true){
    set(adresse);
    _maj=maj;
  }
  virtual ~PseudoXML(){}

  const bool set(const stringpp & adresse){
    f.open(adresse.c_str(),std::ios::in);
    if(!f.good()){
      std::cout<<"Fichier "<<adresse<<" illisible"<<std::endl;
      return false;
    }
    return true;
  }
  const void finir(){
    f.close();
  }
  static const stringpp getnextbalisexml(std::ifstream & f, const bool maj=true){
    //int lmax=200;
    std::string tmp;//char tmp[lmax];
    getline(f,tmp,'>');//f.getline(tmp,lmax,'>');
    stringpp s(tmp);
    s.supprimeavantapres_ep();
    if(s[0]=='<'){s.supprimeavantapres_ep('<');}//Cas de la premiere balise du fichier
    if(maj){s.ToUpCase();}
    return s;
  }
  static const stringpp getnextentrebalisexml(std::ifstream & f){
    //int lmax=200;
    std::string tmp;//char tmp[lmax];
    getline(f,tmp,'<');//f.getline(tmp,lmax,'<');
    stringpp s(tmp);
    s.supprimeavantapres_ep();
    return s;
  }

  static const bool LectureXML(const stringpp & adresse_xml, std::vector<stringpp> & balises, std::vector<stringpp> & contenus, const bool maj=true){
    PseudoXML pxml(maj);
    std::ifstream fxml(adresse_xml.c_str(),std::ios::in);
    if(fxml.fail()){return false;}
    while(!fxml.eof()){
      stringpp nombalise=pxml.getnextbalisexml(fxml,maj);
      stringpp contenubalise=pxml.getnextentrebalisexml(fxml);
      if(nombalise.empty()){continue;}
      //if(nombalise[0]=='/'){continue;}
      balises.push_back(nombalise);
      contenus.push_back(contenubalise);
      //    std::cout<<"Nom balise : "<<nombalise<<" : "<<std::endl;
      //    std::cout<<" Contenu : "<<contenubalise<<std::endl;
    }
    return true;
  }

  static const bool BaliseFin(const stringpp & balise){
    if(balise[0]=='/'){return true;}
    return false;
  }
  static const stringpp BaliseIni2BaliseFin(const stringpp & baliseini){
    return stringpp(stringpp("/")+baliseini);
  }
 

 protected:
  std::ifstream f;
  bool _maj;
};





#endif

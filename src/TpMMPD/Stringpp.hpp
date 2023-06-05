#ifndef _STRINGPP_
#define _STRINGPP_

#include <string>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>

//=============================================================================
// Convertisseur universel...
//mais attention quand m�me � ce qu'on y met
//-----------------------------------------------------------------------------
template<class U, class V>
void ConvertisseurUniversel(const U & valu, V & valv){
  std::stringstream sstr; 
  sstr << valu;
  sstr >> valv;
}


//==============================================================================
//Classe derivee de la classe std::string et comportant quelques methodes supplementaires
//- construction a partir de chiffres
//- formatage pour l'affichage
//- rechercher/remplacer
//- nom de fichers...
//------------------------------------------------------------------------------

class stringpp : public std::string{
 public:
  stringpp():std::string(){}
  stringpp(const char *s):std::string(s){}
  stringpp(const std::string & s):std::string(s){}
  stringpp(const stringpp & s):std::string(s){}
  template<class T> stringpp(const T & val){
    (*this)<<val;
    //std::stringstream sstr; sstr << val; *this = (const std::string &)sstr.str();
  }
  template<class T> stringpp(const T & val, const int precision){
    std::stringstream sstr;
    sstr <<std::fixed<<std::setprecision(precision)<< (const double& )val;
    *this = (const std::string &)sstr.str();
  }
  virtual ~stringpp(){}

  const int atoi() const{return std::atoi((*this).c_str());}
  const double atof() const{
    //return std::atof((*this).c_str());
    //Il arrive qu'atof fasse n'importe quoi
    double value; std::istringstream sstr((*this).c_str()); sstr>>value;
    return value;
  }

  //operateur =
  void operator= (const std::string & s){
    std::string::operator=(s);
  }
  void operator= (const char *s){
    std::string::operator=(s);
  }
  void operator= (const stringpp & s){
    std::string::operator=(s);
  }
  
  //operateur +
  const stringpp operator+ (const std::string & s) const{
    return (const std::string &)(*this)+s;
  }
  const stringpp operator+ (const char *s) const{
    return (const std::string &)(*this)+s;
  }
  const stringpp operator+ (const stringpp & s) const{
    return (const std::string &)(*this)+s;
  }
  template<class T> const stringpp operator+ (const T & val) const{
    //stringpp s(*this); s<<val; return s;
    return (const std::string &)(*this)+stringpp(val);
  }
  
  //operateur +=
  void operator+= (const std::string & s){
    std::string::operator+=(s);
  }
  void operator+= (const char *s){
    std::string::operator+=(s);
  }
  void operator+= (const stringpp & s){
    std::string::operator+=(s);
  }
  template<class T> void operator+= (const T & val){
    (*this)<<val; //(*this)+=stringpp(val);
  }
  template <class T> stringpp & operator<< (const T & val){
    std::stringstream ss;
    ss << val;
    std::string::operator+=(ss.str());
	return (*this);
  }
  /*
  template <class T> stringpp & operator << (const T & val){
    std::stringstream ss;
    ss << val;
    std::string::operator+=(ss.str());
    return (*this);
  }
  */

  stringpp & operator << (const stringpp & s){
    (*this)=(*this)+s;
	return (*this);
  }
  

  void ToLowCase(){
    for(stringpp::iterator i=begin();i!=end();++i){
      (*i)=tolower(*i);
    }
  }
  void ToUpCase(){
    for(stringpp::iterator i=begin();i!=end();++i){
      (*i)=toupper(*i);
    }
  }
  const stringpp LowCase() const{
    stringpp s(*this);
    s.ToLowCase();
    return s;
  } 
  const stringpp UpCase() const{
    stringpp s(*this);
    s.ToUpCase();
    return s;
  } 
  static const char UpCase(char a){return tolower(a);}
  static const char LowCase(char a){return toupper(a);}



  //-------------------------------------------------------------
  //Petits outils


  //Parse suivant un separateur (ex "a,b" --> "a" et "b"
  static void parser(const std::string & aparser, const std::string & separateur, std::vector<std::string> & parties){
    if(aparser.empty()){return;}
    std::string aparsertmp(aparser);
    std::string partie1,partie2;
    std::string::size_type pos=0;
    while(true){
    if(aparsertmp.empty()){break;}
    pos=aparsertmp.find(separateur.c_str());
    if(pos==aparsertmp.npos){//On ne trouve plus de separateur
      //Le dernier element est donc le premier...
      if(!aparsertmp.empty()){parties.push_back(aparsertmp);}
      break;
    }
    partie1=aparsertmp; partie2=aparsertmp;
    partie1.erase(pos,aparsertmp.length());//  partie1.erase(pos,length()-1);
    partie2.erase(0,pos+separateur.size());
    //std::cout<<"'"<<partie1<<"' '"<<partie2<<"'"<<std::endl;
    if(!partie1.empty()){parties.push_back(partie1);}
    aparsertmp=partie2;    
    }
  }

  void parser(const std::string & separateur, std::vector<std::string> & parties) const{
    if(empty()){return;}
    std::string aparser((*this));
    parser(aparser,separateur,parties);
  }
  

  //verifie si la chaine de caractere contient une chaine donnee
  static const bool contient(const std::string & s, const std::string & atrouver){
    if(s.empty()){return false;}
    std::string stmp(s);
    std::string::size_type pos=0;
    pos=stmp.find(atrouver.c_str());
    //std::cout<<"pos "<<(int)pos<<std::endl;
    if(pos==stmp.npos){return false;}
    return true;
  }
  const bool contient(const std::string & atrouver) const{
    return contient((const std::string &)(*this),atrouver);
  }


  //Decoupe en 2 en fonction du premier limitedecoupage rencontr�
  static const bool DecoupePremier(const std::string & adecouper, std::string & partie1, std::string & partie2, const std::string & limitedecoupage){
    stringpp adecoupertmp(adecouper);
    return adecoupertmp.DecoupePremier((stringpp &)partie1,(stringpp &)partie2,(const stringpp &)limitedecoupage);
  }
  const bool DecoupePremier(stringpp & partie1, stringpp & partie2, const stringpp & limitedecoupage) const{
    stringpp::size_type pos=0;
    pos=(*this).find(limitedecoupage.c_str());
    if(pos==(*this).npos){
      partie1="";
      partie2="";
      return false;
    }
    partie1=(*this);
    partie2=(*this);
    partie1.erase(pos,length());//  partie1.erase(pos,length()-1);
    partie2.erase(0,pos+limitedecoupage.size());
    return true;
  }

  //Remplace le premier aremplacer rencontr� par nouveau
  const stringpp RemplacePremier(const stringpp & aremplacer, const stringpp & nouveau="") const{
    stringpp partie1,partie2;
    if(!DecoupePremier(partie1,partie2,aremplacer)){return *this;}
    return partie1+nouveau+partie2;
  }
  //idem mais en place
  void RemplacePremier_ep(const stringpp & aremplacer, const stringpp & nouveau=""){
    (*this)=RemplacePremier(aremplacer,nouveau);
  }
  static const std::string RemplacePremier(const std::string & sini, const std::string & aremplacer, const std::string & nouveau){
    stringpp stmp(sini);
    //stmp=stmp.RemplacePremier(aremplacer,nouveau);
    //return std::string(stmp);
    return (const std::string)stmp.RemplacePremier(aremplacer,nouveau);
  }


 //Suppression des char c avant et apr�s...
  static const stringpp supprimeavantapres(const stringpp & s, const char asupprimer){
    if(s.empty()){return s;}
    std::string::size_type posd,posf;
    posd=s.find_first_not_of(asupprimer);if(posd==std::string::npos){posd=0;}
    posf=s.find_last_not_of(asupprimer);if(posf==std::string::npos){posf=s.length()-1;}
    return s.substr(posd,(int)posf-(int)posd+1);
  }
  //Suppression des elements contenus dans c avant et apr�s...
  static const stringpp supprimeavantapres(const stringpp & s, const stringpp & asupprimer){
    if(s.empty()){return s;}
    std::string::size_type posd,posf;
    posd=s.find_first_not_of(asupprimer);if(posd==std::string::npos){posd=0;}
    posf=s.find_last_not_of(asupprimer);if(posf==std::string::npos){posf=s.length()-1;}
    return s.substr(posd,(int)posf-(int)posd+1);
  }

  const stringpp supprimeavantapres_new(const char asupprimer){
    if((*this).empty()){return (*this);}
    std::string::size_type posd,posf;
    posd=find_first_not_of(asupprimer);if(posd==std::string::npos){posd=0;}
    posf=find_last_not_of(asupprimer);if(posf==std::string::npos){posf=length()-1;}
    return substr(posd,(int)posf-(int)posd+1);
  }
  //Suppression des elements contenus dans c avant et apr�s...
  const stringpp supprimeavantapres_new(const stringpp & asupprimer){
    if(empty()){return (*this);}
    std::string::size_type posd,posf;
    posd=find_first_not_of(asupprimer);if(posd==std::string::npos){posd=0;}
    posf=find_last_not_of(asupprimer);if(posf==std::string::npos){posf=length()-1;}
    return substr(posd,(int)posf-(int)posd+1);
  }
  void supprimeavantapres_ep(const char asupprimer){(*this)=supprimeavantapres_new(asupprimer);}
  void supprimeavantapres_ep(const stringpp & asupprimer){(*this)=supprimeavantapres_new(asupprimer);}
    
  
  //Suppression des espaces ' ','\n','\t' avant et apr�s...
  const stringpp supprimeavantapres_new() const{
    if((*this).empty()){return *this;}
    std::string::size_type posd,posf;
    std::string asupprimer(" \n\t");
    posd=(*this).find_first_not_of(asupprimer);if(posd==std::string::npos){posd=0;}
    posf=(*this).find_last_not_of(asupprimer);if(posf==std::string::npos){posf=(*this).length()-1;}
    return (*this).substr(posd,(int)posf-(int)posd+1);
  }

  //Suppression des espaces ' ','\n','\t' avant et apr�s...
  void supprimeavantapres_ep(){
    (*this)=supprimeavantapres_new();
  }

  //Suppression des espaces ' ','\n','\t' intermediaires...
  const stringpp supprimeespacesintermediaires_new() const{
    std::istringstream iss((*this).c_str());
    stringpp final;
    while(!iss.eof()){
      stringpp stmp;
      iss>>stmp;
      if(stmp.empty()){continue;}
      final=final+stmp;
    }
    return final;
  }

  void supprimeespacesintermediaires_ep(){
    (*this)=supprimeespacesintermediaires_new();
  }

  const stringpp remplaceespacesintermediaires_new(const std::string & caractere_de_remplacement) const{
    std::istringstream iss((*this).c_str());
    stringpp final;
    while(!iss.eof()){
      stringpp stmp;
      iss>>stmp;
      if(stmp.empty()){continue;}
      if(final.empty()){final=stmp;}else{final=final+caractere_de_remplacement+stmp;}
    }
    return final;
  }

  void remplaceespacesintermediaires_ep(const std::string & caractere_de_remplacement){
    (*this)=supprimeespacesintermediaires_new();
  }

  ///------------------------------------------------------------
  //Partie namespace : contient des outils de manipulation des strings


  //    Convertisseur universel... mais attention quand m�me � ce qu'on y met
  template<class U, class V>
  static void ConvertisseurUniversel(const U & valu, V & valv){
    std::stringstream sstr; 
    sstr << valu;
    sstr >> valv;
  }

  static const std::string int2string(const int i){
    std::stringstream sstr;
    sstr<<i;
    return sstr.str();
  }
  static const std::string double2string(const double d){
    std::stringstream sstr;
    sstr<<d;
    return sstr.str();
  }
  static const std::string double2string(double d, int precision){
    std::stringstream sstr;
    sstr <<std::fixed<<std::setprecision(precision)<<d;
    return sstr.str();
  }
  template<class T> const static std::string ToString(const T & val){
    std::stringstream sstr;
    sstr<<val;
    return sstr.str();
  }

  static const int string2int(const std::string & str){return std::atoi(str.c_str());} 
  static const double atoi(const std::string & str){return string2int(str);}
  static const double string2double(const std::string & str){
    //return std::atof(str.c_str());
    //Il arrive qu'atof fasse n'importe quoi
    double value; std::istringstream sstr(str.c_str()); sstr>>value;
    return value;
  }
  static const double atof(const std::string & str){return string2double(str);}


  static void ToLowCase(std::string & s){
    for(std::string::iterator i=s.begin();i!=s.end();++i){
      (*i)=tolower(*i);
    }
  }
  static void ToUpCase(std::string & s){
    for(std::string::iterator i=s.begin();i!=s.end();++i){
      (*i)=toupper(*i);
    }
  }  
  static const std::string LowCase(const std::string & s){
    std::string str(s);
    ToLowCase(str);
    return str;
  }
  static const std::string UpCase(const std::string & s){
    std::string str(s);
    ToUpCase(str);
    return str;
  }  


  //--------------------------------------------------------------------
  //Outils de manipulation des noms de fichiers

  //Teste la validite du fichier situe a l'adresse (*this)
  const bool FileExist() const{
    return std::ifstream((*this).c_str()).good();
  }
  const bool existence_fichier() const{
    return FileExist();
  }

  const stringpp GetExtension() const{
    return GetExtension((*this));
  }
  void SetExtension(const stringpp & nouvelle_extension){
    SetExtension(*this,nouvelle_extension);
  }
  const stringpp newSetExtension(const stringpp & nouvelle_extension) const{
    return newSetExtension(*this,nouvelle_extension);
  }
  void SetExtension(const char *nouvelle_extension){
    SetExtension(*this,nouvelle_extension);
  }
  const stringpp newSetExtension(const char *nouvelle_extension) const{
    return newSetExtension(*this,nouvelle_extension);
  }
  const stringpp Extension() const{
    return Extension(*this);
  }
  void SupprExtension(){
    SupprExtension(*this);
  }
  const stringpp newSupprExtension() const{
    return newSupprExtension(*this);
  }
  void adressefromDirFile(const stringpp & dir, const stringpp & file){
    (*this)=AdressefromDirFile(dir,file);
  }
  void DirFile(stringpp &dir, stringpp &file) const{
    DirFile(*this,dir,file);
  }
  const stringpp File() const{
    return File(*this);
  }
  const stringpp Dir() const{
    return Dir(*this);
  }
  void CombineNoms(const stringpp & adresse1, const stringpp & adresse2, const stringpp & nouvelle_extension=""){
    (*this)=CombineNom(adresse1,adresse2,nouvelle_extension);
  }
  void DeCombineNoms(stringpp & adresse1, stringpp & adresse2, const stringpp & nouvelle_extension="") const{
    DeCombineNom(*this,adresse1,adresse2,nouvelle_extension);
  }



  //-------------------------------------------------------------
  //namespace
  
  static const stringpp GetExtension(const stringpp & adresse){
    stringpp ext(adresse);
    stringpp::size_type pos=0;
    pos=ext.find_last_of(".");
    if(pos!=ext.npos){
      ext.erase(0,pos+1);//pour ne pas garder de point
    }  
    return ext;
  }

  static void SetExtension(stringpp & adresse, const stringpp & nouvelle_extension){
    if(nouvelle_extension==stringpp("")){SupprExtension(adresse);return;}
    else if(nouvelle_extension==stringpp(".")){SupprExtension(adresse);adresse+=stringpp(".");return;}
    stringpp::size_type pos=0;
    pos=adresse.find_last_of(".");
    //    std::cout<<pos<<" "<<adresse.npos<<std::endl;
    if(pos!=adresse.npos && pos!=0){//si pos==0 on est sans dout dans un cas ./nomfichier. ; si pos=npos, il n'y a vraiment pas de .
      adresse.erase(pos,adresse.length());//      adresse.erase(pos,adresse.length()-1);
    }
    //  if(nouvelle_extension.find_last_of(".")==nouvelle_extension.npos){
    if(nouvelle_extension.find(".")!=0){
      adresse=adresse+stringpp(".");
    }
    adresse=adresse+nouvelle_extension;
  }

  static const stringpp newSetExtension(const stringpp & adresse, const stringpp & nouvelle_extension){  
    stringpp nouvelle_adresse(adresse);
    SetExtension(nouvelle_adresse,nouvelle_extension);
    return nouvelle_adresse;
  }

  static void SetExtension(stringpp & adresse, const char *nouvelle_extension){
    SetExtension(adresse,stringpp(nouvelle_extension));
  }

  static const stringpp newSetExtension(const stringpp & adresse, const char *nouvelle_extension){
    return newSetExtension(adresse,stringpp(nouvelle_extension));
  }

  static const stringpp Extension(const stringpp & adresse){
    stringpp ext(adresse);
    stringpp::size_type pos=0;
    pos=ext.find_last_of(".");
    if(pos!=ext.npos){
      //    ext.erase(0,pos);//pour garder un point
      ext.erase(0,pos+1);//pour ne pas garder de point
      return ext;
    }else{
      return stringpp("");
    }
  }


  static void SupprExtension(stringpp & adresse){
    stringpp::size_type pos=0;
    pos=adresse.find_last_of(".");
    if(pos!=adresse.npos){
      adresse.erase(pos,adresse.length());//pour ne pas garder de point
    }
  }

  static const stringpp newSupprExtension(const stringpp & adresse){
    stringpp newadresse(adresse);
    SupprExtension(newadresse);
    return newadresse;
  }
  
  static const stringpp AdressefromDirFile(const stringpp & dir, const stringpp & file){
    stringpp dirfile=dir+file;
    stringpp::size_type pos=0;
    pos=dir.find_last_of("/");
    if(pos!=dir.length()-1){//  if(pos==dir.npos){
      pos=dir.find_last_of("\\");
    }
    if(pos!=dir.length()-1){//  if(pos==dir.npos){
      dirfile=dir+stringpp("/")+file;
    }
    dirfile.RemplacePremier_ep("//","/");//Il s'agit de supprimer d'eventuels '//' causes par la concatenation...
    return dirfile;
  }

  static void DirFile(const stringpp & adresse, stringpp &dir, stringpp &file){
    stringpp::size_type pos=0;
    pos=adresse.find_last_of("/");
    if(pos==adresse.npos){
      pos=adresse.find_last_of("\\");
    }
    if(pos==adresse.npos){
      file=adresse;
      dir=stringpp("");
    }else{
      file=adresse;
      dir=adresse;
      file.erase(0,pos+1);
      dir.erase(pos,adresse.length()-1);
    }
  }

  static const stringpp File(const stringpp & adresse){
    stringpp file;
    stringpp::size_type pos=0;
    pos=adresse.find_last_of("/");
    if(pos==adresse.npos){
      pos=adresse.find_last_of("\\");
    }
    if(pos==adresse.npos){
      file=adresse;
    }else{
      file=adresse;
      file.erase(0,pos+1);
    }
    return file;
  }

  static const stringpp Dir(const stringpp & adresse){
    stringpp dir;
    stringpp::size_type pos=0;
    pos=adresse.find_last_of("/");
    if(pos==adresse.npos){
      pos=adresse.find_last_of("\\");
    }
    if(pos==adresse.npos){
      dir=stringpp("");
    }else{
      dir=adresse;
      dir.erase(pos,adresse.length()-1);
    }
    return dir;
  }

  static const stringpp CombineNom(const stringpp & adresse1, const stringpp & adresse2, const stringpp & nouvelle_extension=""){
    stringpp adresse=adresse1;
    stringpp fich2;
    fich2=File(adresse2);
    SetExtension(adresse,".-.");
    if(nouvelle_extension!=""){
      SetExtension(fich2,nouvelle_extension);
    }
    return adresse+fich2;
  }

  static void DeCombineNom(const stringpp & adresse, stringpp & adresse1, stringpp & adresse2, const stringpp & nouvelle_extension=""){
    stringpp adressetmp(File(adresse));
    SupprExtension(adressetmp);
    stringpp::size_type pos=0;
    pos=adressetmp.find(".-.");
    if(pos==adressetmp.npos){
      adresse1="";
      adresse2="";
      return;
    }
    adresse1=adressetmp;
    adresse2=adressetmp;
    adresse1.erase(pos,adresse.length()-1);//  adresse1.erase(pos+1,adresse.length()-1);//ancienne version : on ajoutait un . a la fin du nom
    adresse2.erase(0,pos+3);
    //  adresse2=adresse2+".";//ancienne version : on ajoutait un . a la fin du nom
    if(nouvelle_extension!=stringpp("")){
      SetExtension(adresse1,nouvelle_extension);
      SetExtension(adresse2,nouvelle_extension);
    }
  }


};



#endif 


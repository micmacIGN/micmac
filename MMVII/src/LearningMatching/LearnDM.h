#include "include/MMVII_all.h"

/*
    Caracteristiques envisagees :

      * L1  sur tout pixel
      * L1  pondere par 1/Scale sur tout pixel
      * L1 sur le + petit rayon
      * census (pondere ?)
      * correlation  (pondere ? 1 pixel? 2 pixel ? ...)
      * gradient de L1  sur paralaxe (mesure ambiguite residuelle)
      * mesure sur gradient (rho ? theta ? ...)
*/

namespace MMVII
{

class cNameFormatTDEDM 
{
    protected :
        static inline std::string  PrefixAll()   {return "DMTrain_";}

        static inline std::string  Im1()   {return "Im1";}
        static inline std::string  Im2()   {return "Im2";}
        static inline std::string  Px1()   {return "Pax1";}
        static inline std::string  Px2()   {return "Pax2";}
        static inline std::string  Masq1() {return "Masq1";}
        static inline std::string  Masq2() {return "Masq2";}

        static inline std::string MakeName(const std::string & aName,const std::string & aPref) 
        {
             return PrefixAll() + aName + "_" + aPref + ".tif";
        }

        static inline void GenConvertIm(const std::string & aInput, const std::string & aOutput)
        {
            std::string aCom =   "convert -colorspace Gray -compress none " + aInput + " " + aOutput;
            GlobSysCall(aCom);
        }

        static inline std::string NameIm1(const std::string & aName) {return MakeName(aName,Im1());}
        static inline std::string NameIm2(const std::string & aName) {return MakeName(aName,Im2());}
        static inline std::string NamePx1(const std::string & aName) {return MakeName(aName,Px1());}
        static inline std::string NamePx2(const std::string & aName) {return MakeName(aName,Px2());}
        static inline std::string NameMasq1(const std::string & aName) {return MakeName(aName,Masq1());}
        static inline std::string NameMasq2(const std::string & aName) {return MakeName(aName,Masq2());}

        static inline std::string NameRedrIm1(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im1());}
        static inline std::string NameRedrIm2(const std::string & aName) {return MakeName(aName,"REDRIn_"+Im2());}

        static inline void ConvertIm1(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm1(aName));}
        static inline void ConvertIm2(const std::string & aInput,const std::string & aName) {GenConvertIm(aInput,NameIm2(aName));}


        static inline std::string Im2FromIm1(const std::string & aIm1)
        {
             return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Im2()+".tif");
        }
        static inline std::string Px1FromIm1(const std::string & aIm1)
        {
             return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Px1()+".tif");
        }
        static inline std::string Masq1FromIm1(const std::string & aIm1)
        {
             return replaceFirstOccurrence(aIm1,"_"+Im1()+".tif","_"+Masq1()+".tif");
        }
};

class cVecCaracMatch
{
     public :
        void SetValue(eModeCaracMatch aCarac,const float & aVal) {mVecCarac.at(int(aCarac)) = aVal;}
        const float & Value(eModeCaracMatch aCarac) const ;

        cVecCaracMatch
        (
             float aScaleRho,float aGrayLev1,float aGrayLev2,
             const cAimePCar &,const cAimePCar &
       );
     private :
        static constexpr float UnDefVal = -1e10;
        cVecCaracMatch() ;
        std::vector<float>  mVecCarac;
};

};


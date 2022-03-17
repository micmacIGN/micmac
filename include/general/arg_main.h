/*Header-MicMac-eLiSe-25/06/2007

    MicMac : Multi Image Correspondances par Methodes Automatiques de Correlation
    eLiSe  : ELements of an Image Software Environnement

    www.micmac.ign.fr


    Copyright : Institut Geographique National
    Author : Marc Pierrot Deseilligny
    Contributors : Gregoire Maillet, Didier Boldo.

[1] M. Pierrot-Deseilligny, N. Paparoditis.
    "A multiresolution and optimization-based image matching approach:
    An application to surface reconstruction from SPOT5-HRS stereo imagery."
    In IAPRS vol XXXVI-1/W41 in ISPRS Workshop On Topographic Mapping From Space
    (With Special Emphasis on Small Satellites), Ankara, Turquie, 02-2006.

[2] M. Pierrot-Deseilligny, "MicMac, un lociel de mise en correspondance
    d'images, adapte au contexte geograhique" to appears in
    Bulletin d'information de l'Institut Geographique National, 2007.

Francais :

   MicMac est un logiciel de mise en correspondance d'image adapte
   au contexte de recherche en information geographique. Il s'appuie sur
   la bibliotheque de manipulation d'image eLiSe. Il est distibue sous la
   licences Cecill-B.  Voir en bas de fichier et  http://www.cecill.info.


English :

    MicMac is an open source software specialized in image matching
    for research in geographic information. MicMac is built on the
    eLiSe image library. MicMac is governed by the  "Cecill-B licence".
    See below and http://www.cecill.info.

Header-MicMac-eLiSe-25/06/2007*/
/*eLiSe06/05/99

     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

        eLiSe : Elements of a Linux Image Software Environment

        This program is free software; you can redistribute it and/or modify
        it under the terms of the GNU General Public License as published by
        the Free Software Foundation; either version 2 of the License, or
        (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to the Free Software
        Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

          Author: Marc PIERROT DESEILLIGNY    IGN/MATIS
          Internet: Marc.Pierrot-Deseilligny@ign.fr
             Phone: (33) 01 43 98 81 28
*/

#ifndef _ELISE_GENERAL_ARG_MAIN_H
#define _ELISE_GENERAL_ARG_MAIN_H

#if ElMemberTpl

//#include <strstream>

#include "CMake_defines.h"

class cMMSpecArg;

std::string MakeStrFromArgcARgv( int argc, char **argv, bool aProtect = false);

void MemoArg(int,char**);
void ShowArgs();

extern std::string TheStringMemoArgOptGlob;
extern std::string GlobArcArgv;



// Remet dans la commande les option commancant par "-"
void AddMinuToArgs(std::string & aCom,int  argc,char** argv);

class ElGramArgMain  // classe contenant la "grammaire" rudimentaire
{
    public :
        ElGramArgMain(char,int,char,bool aAnyEqual);


        const char  mCharEq;
        const int   mCharBeginTab;
        const char  mCharEndTab;
                bool  AnyEqual() const;

                static  const ElGramArgMain  StdGram;
                static  const ElGramArgMain  SPHGram;
                static  const ElGramArgMain  THOMGram;
                static  const ElGramArgMain  HDRGram;
        private :
                bool        mAnyEqual;
};



template <class Type> inline std::istream &  ElStdRead (std::istream &is,Type & obj,const ElGramArgMain &)
{
    return is >> obj;
}

inline std::istream &  ElStdRead(std::istream &is,std::string &obj, const ElGramArgMain &)
{
    //return is >> obj;
    return getline(is,obj);
}

extern bool Str2Bool(bool & aRes,const std::string & aStr);
extern bool Str2BoolForce(const std::string & aStr);

template <> inline std::istream & ElStdRead (std::istream &is, bool & aVal, const ElGramArgMain & /*G*/)
{
   std::string aStr ;
   is >> aStr;
   aVal = Str2BoolForce(aStr);
   return is;
}


// template <class Type>  std::istream & operator >> (std::istream &is,ElSTDNS vector<Type> & vec);

template <class Type>  inline std::istream & VElStdRead (std::istream &is,ElSTDNS vector<Type> & vec, const ElGramArgMain & Gram)
{
    vec.clear();

    if (Gram.mCharBeginTab != -1)
    {
       if (is.get() != Gram.mCharBeginTab)
          ELISE_ASSERT(false,"istream >> vector<Type>");
    }

    int c;
    while ((c = is.get()) !=   Gram.mCharEndTab)
    {
        ELISE_ASSERT (c!=-1,"Unexpected End Of String in ElStdRead(vector<Type> &)");
        if (c!=',')
           is.unget();
        Type v;
        is >> v;
        vec.push_back(v);
    }
    return is;
}

#define SPECIALIZE_ElStdRead(aTYPE)\
template <> inline std::istream & ElStdRead (std::istream &is, ElSTDNS vector < aTYPE > & vec, const ElGramArgMain & G)\
{\
    return VElStdRead(is,vec,G);\
}

SPECIALIZE_ElStdRead (INT)
SPECIALIZE_ElStdRead (ElSTDNS vector <INT>)
SPECIALIZE_ElStdRead (REAL)
SPECIALIZE_ElStdRead (Pt2dr)


std::istream & VStrElStdRead
                      (
                              std::istream &is,
                              ElSTDNS vector<std::string> & vec,
                              const ElGramArgMain & Gram
                      );


template <> inline std::istream & ElStdRead
                                  (
                                        std::istream &is,
                                        ElSTDNS vector <std::string > & vec,
                                        const ElGramArgMain & G
                                  )
{
    return VStrElStdRead(is,vec,G);
}

template <class Type> std::ostream & operator << (std::ostream &os,const ElSTDNS vector<Type> & v)
{
    os << "[";
    for (INT k=0; k<(INT)v.size(); k++)
    {
        if (k!=0) os<< ",";
        os << v[k];
    }
    os << "]";
    return os;
}


typedef enum
{
    eSAM_None,
    eSAM_IsBool,
    eSAM_IsPowerOf2,
    eSAM_IsDir,
    eSAM_IsPatFile,
    eSAM_IsExistDirOri,
    eSAM_IsOutputDirOri,
    eSAM_IsExistFile,
    eSAM_IsExistFileRP, //Relative path
    eSAM_IsOutputFile,
    eSAM_Normalize,
    eSAM_NoInit,
    eSAM_InternalUse
} eSpecArgMain;

typedef enum
{
    AMBT_Box2di,
    AMBT_Box2dr,
    AMBT_bool,
    AMBT_INT,
    AMBT_REAL,
    AMBT_Pt2di,
    AMBT_Pt2dr,
    AMBT_Pt3dr,
    AMBT_Pt3di,
    AMBT_string,
    AMBT_U_INT1,
    AMBT_INT1,
    AMBT_char,
    AMBT_vector_Pt2dr,
    AMBT_vector_int,
    AMBT_vector_double,
    AMBT_vvector_int,
    AMBT_vector_string,
    AMBT_unknown
} eArgMainBaseType;

class GenElArgMain
{
    public :
        virtual ~GenElArgMain()  {};
            GenElArgMain
                (
                      const char * Name,
                      bool ISINIT,
                      eSpecArgMain aSpec,
                      const std::list<std::string> & aLEnumVal
                ) ;
        virtual GenElArgMain * dup() const = 0;

        virtual void InitEAM(const ElSTDNS string &s,const ElGramArgMain &) const = 0;
        virtual void * AddrArg() const = 0;
        bool InitIfMatchEq(const ElSTDNS string &s,const ElGramArgMain & Gram) const;

        bool IsInit() const;
        const char *name() const;

        virtual void show(bool named) const =0;

          // Ensemble de patch pour rajouter des arguments inutilises a la volee
        bool IsActif() const;

        static const char * ActifStr(bool);

        virtual std::string NameType() const =0;
        virtual std::string Comment() const =0;
        eSpecArgMain Spec() const;
        const std::list<std::string> &  ListEnum() const;

        virtual eArgMainBaseType type() const { return AMBT_unknown; }

    protected :
                static const std::string  theChaineInactif;
                static const std::string  theChaineActif;

        ElSTDNS string  _name;
        mutable bool  _is_init;
                eSpecArgMain    mSpec;
                std::list<std::string>  mLEnum;
};


template <class Type> const char * str_type(Type *);


extern std::set<void *>  AllAddrEAM;
extern std::list<std::string>  TheEmptyListEnum;
extern std::map<void *,std::string>  MapValuesEAM;



std::list<std::string> ModifListe(const std::list<std::string> &,const char * aNameType);


template <class Type> class ElArgMain : public GenElArgMain
{
    public :
                std::string NameType() const {return  str_type(_adr);}
                std::string Comment() const {return  mCom;}
                Type*       DefVal() const {return _adr;}

        virtual void * AddrArg() const  {return _adr;}
        void InitEAM(const ElSTDNS string &s,const ElGramArgMain & Gram) const
        {
                        AllAddrEAM.insert( (void *) _adr);
            _is_init = true;
            std::STD_INPUT_STRING_STREAM Is(s.c_str());
            // Is >> *_adr;
            ::ElStdRead(Is,*_adr,Gram);
        }

         ElArgMain
             (
                   Type & v,
                   const char * Name,
                   bool isInit,
                   const std::string & aCom = "",
                   eSpecArgMain        aSpec =  eSAM_None,
                   const std::list<std::string> & aLEnumVal = TheEmptyListEnum

             ) :
            GenElArgMain(Name,isInit,aSpec,ModifListe(aLEnumVal,str_type((Type*)0))),
            _adr   (&v),
                        mCom   (aCom)
         {
         }

        GenElArgMain * dup() const
        {
            return new ElArgMain<Type> (*this);
        }
                void show(bool named) const
                {
                    std::cout << "  * ";
                    if (named)
                       std::cout << "[Name=" << name() <<"] " ;

                    std::cout << str_type(_adr);
                    if (mCom != "")
                       std::cout << " :: {" << mCom  <<"}" ;
                    std::cout <<"\n";
                }


             static const ElSTDNS list<Type>  theEmptyLvalADM;

             eArgMainBaseType type() const;

    private :

        Type *      _adr;
                std::string  mCom;

};

bool EAMIsInit(void *);
extern std::string StrInitOfEAM(void * anAdr) ;

std::string StrFromArgMain(const std::string & aStr);

/*
template <> inline void ElArgMain<std::string>::InitEAM(const ElSTDNS string &s,const ElGramArgMain & Gram) const
{
   _is_init = true;
   *_adr = StrFromArgMain(s);
   AllAddrEAM.insert( (void *) _adr);
}
*/


template <class Type> ElArgMain<Type>
                     EAM
                     (
                            Type & v,
                            const char * Name= "",
                            bool isInit = false,
                            const std::string &aComment = "",
                            eSpecArgMain        aSpec =  eSAM_None,
                            const std::list<std::string> & aLEnumVal = TheEmptyListEnum
                     )
{
        return ElArgMain<Type>(v,Name,isInit,aComment,aSpec,aLEnumVal);
}
template <class Type> ElArgMain<Type>
                     EAMC
                     (
                            Type & v,
                            const std::string &aComment ,
                            eSpecArgMain        aSpec =  eSAM_None,
                            const std::list<std::string> & aLEnumVal = TheEmptyListEnum
                     )
{
                AllAddrEAM.insert( (void *) &v);
        return ElArgMain<Type>(v,"",false,aComment,aSpec,aLEnumVal);
}



class LArgMain
{
    public :

        std::vector<cMMSpecArg>  ExportMMSpec(bool isOpt = false) const;

        template <class Type> LArgMain & operator << (const ElArgMain<Type> & v)
        {
             if (v.IsActif())
               _larg.push_back(v.dup());
            return * this;
        }
        ~LArgMain();
        LArgMain & operator << (const LArgMain  & v);

        int Size() const;

        INT  Init(int argc,char ** argv) const;
                void  InitIfMatchEq
                      (
                          std::vector<char *> *,  // Si !=0, empile les args non consommes
                          int argc,char ** argv,const ElGramArgMain & Gram,
                          bool VerifInit=true,bool AccUnK=false
                      ) const;

        void show(bool named) const;

        LArgMain();
        void VerifInitialize() const;

        bool OneInitIfMatchEq
                     (
                          char *,
                          const ElGramArgMain & Gram,
                          bool  anAcceptUnknown
                     ) const;
    private :
        ElSTDNS list<GenElArgMain *> _larg;
                // Apparemment certains compilos
                // utilisent la copie en temporaire;
        //      LArgMain(const LArgMain &);
        // void operator = (const LArgMain &);
};





// Renvoie eventuellement la partie non consommee
#define EIAM_VerifInit true
#define EIAM_AccUnK false
#define EIAM_NbArgGlobGlob -1

// Var glob, rajoutee pour indiquer que MICMAC est en mode visuel
// initialisee dans GenMain, utilisee dans ElInitArgMain
extern bool MMVisualMode;

// Ch.M: MMRunVisualMode is now a pointer to function which is statically
// initialized to the empty function MMRunVisualModeNoQt (return 0).
// mm3d may dynamically changes it to MMRunVisualModeQt if needed.
// This trick allows to put the depandancy on Qt library only in mm3d
// instead of all users of ElInitArgMain
extern int (*MMRunVisualMode)
     (
         int argc,char ** argv,
         std::vector<cMMSpecArg> & aVAM,
         std::vector<cMMSpecArg> & aVAO,
         std::string aFirstArg
     );

int MMRunVisualModeQt
     (
         int argc,char ** argv,
         std::vector<cMMSpecArg> & aVAM,
         std::vector<cMMSpecArg> & aVAO,
         std::string aFirstArg = ""
     );

int MMRunVisualModeNoQt
     (
         int argc,char ** argv,
         std::vector<cMMSpecArg> & aVAM,
         std::vector<cMMSpecArg> & aVAO,
         std::string aFirstArg = ""
     );


typedef void (*tActionOnHelp)(int argc,char ** argv);
extern tActionOnHelp TheActionOnHelp;


std::vector<char *>     ElInitArgMain
        (
            int argc,char ** argv,
            const LArgMain & ,
            const LArgMain & ,
            const std::string & aFirstArg = "",
            bool  VerifInit=EIAM_VerifInit,
            bool  AccUnK=EIAM_AccUnK,
            int   aNbArgGlobGlob = EIAM_NbArgGlobGlob
        );

void    ElInitArgMain
        (
            const std::string &,
            const LArgMain & ,
            const LArgMain & ,
            const std::string & aArg = ""
        );


void SphInitArgs(const ElSTDNS string & NameFile,const LArgMain &);
void StdInitArgsFromFile(const ElSTDNS string & NameFile,const LArgMain &);
void HdrInitArgsFromFile(const ElSTDNS string & NameFile,const LArgMain &);
INT ThomInitArgs(const ElSTDNS string & NameFile,const LArgMain &);
bool IsThomFile (const std::string & aName);


class cReadObject;
typedef const char * tCharPtr;

class cReadObject
{
     public :



        bool  Decode(const char * aLine);
        static bool ReadFormat(char  & aCom,std::string & aFormat,const std::string &aFileOrLine,bool IsFile);

        double GetDef(const double & aVal,const double & aDef);
        Pt3dr  GetDef(const Pt3dr  & aVal,const double  & aDef);
        std::string  GetDef(const std::string  & aVal,const std::string  & aDef);

        bool IsDef(const double &) const;
        bool IsDef(const Pt3dr &) const;
        bool IsDef(const std::string &) const;

     protected :
         std::string GetNextStr(tCharPtr &);


         cReadObject(char aComCar,const std::string & aFormat, const std::string & aSymbUnknown);
         void VerifSymb(const std::string &aS,bool Required);
         void AddDouble(const std::string & aS,double * anAdr,bool Required);
         void AddDouble(char aC,double * anAdr,bool Required);
         void AddPt3dr(const std::string & aS,Pt3dr * aP,bool Required);
         void AddPt2dr(const std::string & aS,Pt2dr * aP,bool Required);
         void AddString(const std::string & aS,std::string * aName,bool Required);



         char                                   mComC;
         std::string                            mFormat;
         std::string                            mSymbUnknown;
         std::set<std::string>                  mSymbs;
         std::set<std::string>                  mSFormat;
         std::map<std::string,double *>         mDoubleLec;
         std::map<std::string,std::string *>    mStrLec;
         static const double TheDUnDef;
         static const std::string TheStrUnDef;
         int mNumLine;
};

// uti_files
int BatchFDC_main(int argc,char ** argv);
int MapCmd_main(int argc,char ** argv);
int MyRename_main(int argc,char ** argv);
int Genere_Header_TiffFile_main(int argc,char ** argv);
int TestSet_main(int argc,char ** argv);
int TestMTD_main(int argc,char ** argv);
int TestCmds_main(int argc,char ** argv);
int Apero2PMVS_main(int argc, char ** argv);
int Apero2Meshlab_main(int argc, char ** argv);
int Ori2XML_main(int argc,char ** argv);
int GenCode_main(int argc,char ** argv);
int XifGps2Xml_main(int argc,char ** argv);
int XifGps2RTL_main(int argc,char ** argv);


// uti_images
int Undist_main(int argc,char ** argv);
int Dequant_main(int argc,char ** argv);
int Devlop_main(int argc,char ** argv);
int TiffDev_main(int argc,char ** argv);
int ElDcraw_main(int argc,char ** argv);
int GenXML2Cpp_main(int argc,char ** argv);
int GrShade_main(int argc,char ** argv);
int LumRas_main(int argc,char ** argv);

int DevOneImPtsCarVideo_main(int argc,char ** argv);
int Devideo_main(int argc,char ** argv);




int CoherEpi_main(int argc,char ** argv);
int BlockCoherEpi_main(int argc,char ** argv);

int EstimFlatField_main(int argc,char ** argv);
int Impainting_main(int argc,char ** argv);
int MpDcraw_main(int argc,char ** argv);
int PastDevlop_main(int argc,char ** argv);
int Reduc2MM_main(int argc,char ** argv);
int ScaleIm_main(int argc,char ** argv);
int StatIm_main(int argc,char ** argv);
int ConvertIm_main(int argc,char ** argv);
int MakePlancheImage_main(int argc,char ** argv);
int tiff_info_main(int argc,char ** argv);
int to8Bits_main(int argc,char ** argv);
int mmxv_main(int argc,char ** argv);
int MPDtest_main(int argc,char ** argv);
int CmpIm_main(int argc,char ** argv);
int Drunk_main(int argc,char ** argv);
int CalcSzWCor_main(int argc,char ** argv);
int Digeo_main(int argc,char ** argv);
int Vignette_main(int argc,char ** argv);
int Arsenic_main(int argc,char ** argv);
int Sift_main(int argc,char ** argv);
int Ann_main(int argc,char ** argv);
int GenMire_main (int argc,char** argv);
int GrayTexture_main (int argc,char** argv);
int SplitMPO_main(int argc,char ** argv);

// uti_phgram
int AperiCloud_main(int argc,char ** argv);
int Apero_main(int argc,char ** argv);
int Bascule_main(int argc,char ** argv);
int CmpCalib_main(int argc,char ** argv);
int ConvertCalib_main(int argc, char** argv);
int Campari_main(int argc,char ** argv);
int CASA_main(int argc,char ** argv);
int Donuts_main(int argc,char **argv);
int MMTestOrient_main(int argc,char ** argv);
int MMHomCorOri_main(int argc,char ** argv);
int ChgSysCo_main(int argc,char ** argv);
int GCPBascule_main(int argc,char ** argv);
int CentreBascule_main(int argc,char ** argv);
int MakeGrid_main(int argc,char ** argv);
int Malt_main(int argc,char ** argv);
int MMByPair_main(int argc,char ** argv);
int MMOnePair_main(int argc,char ** argv);
int MMSymMasqAR_main(int argc,char ** argv);
int ChantierClip_main(int argc,char ** argv);
int ClipIm_main(int argc,char ** argv);

int GetP3d_main(int argc,char ** argv);
int MergePly_main(int argc,char ** argv);
int MICMAC_main(int argc,char ** argv);
int FusionCarteProf_main(int argc,char ** argv);
int Nuage2Ply_main(int argc,char ** argv);
int Nuage2Homol_main(int argc,char ** argv);
int Txt2Dat_main(int argc,char ** argv);
int PlySphere_main(int argc,char ** argv);
int San2Ply_main(int argc,char ** argv);


int Pasta_main(int argc,char ** argv);
int Pastis_main(int argc,char ** argv);
int Porto_main(int argc,char ** argv);
int Prep4masq_main(int argc,char ** argv);
int ReducHom_main(int argc,char ** argv);
int RHH_main(int argc,char ** argv);
int RHHComputHom_main(int argc,char ** argv);
int MakeOneXmlXifInfo_main(int argc,char ** argv);
int Xml2Dmp_main(int argc,char ** argv);
int Dmp2Xml_main(int argc,char ** argv);


int Morito_main(int argc,char ** argv);
int Liquor_main(int argc,char ** argv);
int Luxor_main(int argc,char ** argv);



int RepLocBascule_main(int argc,char ** argv);
int SBGlobBascule_main(int argc,char ** argv);
int HomFilterMasq_main(int argc,char ** argv);
int Tapas_main(int argc,char ** argv);
int Tapioca_main(int argc,char ** argv);
int Tarama_main(int argc,char ** argv);
int Tawny_main(int argc,char ** argv);
int Tequila_main(int argc,char ** argv);
int TestCam_main(int argc,char ** argv);
int TestChantier_main(int argc,char ** argv);
int TiPunch_main(int argc,char ** argv);
int ScaleNuage_main(int argc,char ** argv);
int SysCoordPolyn_main(int argc,char ** argv);
int Gri2Bin_main(int argc,char ** argv);
int XYZ2Im_main(int argc,char ** argv);
int Im2XYZ_main(int argc,char ** argv);

int SupMntIm_main(int argc,char ** argv);
int ChamVec3D_main(int argc,char ** argv);
int SampleLibElise_main(int argc,char ** argv);



int MMPyram_main(int argc,char ** argv);
int ReechInvEpip_main(int argc,char ** argv);
int CreateEpip_main(int argc,char ** argv);
int AperoChImMM_main(int argc,char ** argv);
int MM_FusionNuage_main(int argc,char ** argv);
int MMInitialModel_main(int argc,char ** argv);
int MMAllAuto_main(int argc,char ** argv);
int MM2DPostSism_Main(int argc,char ** argv);
int CheckDependencies_main(int argc,char ** argv);
int NuageBascule_main(int argc,char ** argv);
int  cod_main(int argc,char ** argv);
int  vicod_main(int argc,char ** argv);
int  genmail_main(int argc,char ** argv);
int Ori_Txt2Xml_main(int argc,char ** argv);
int OriExport_main(int argc,char ** argv);
int GCP_Txt2Xml_main(int argc,char ** argv);
int VideoVisage_main(int argc,char ** argv);
//int Poisson_main(int argc,char ** argv);
int GrapheHom_main(int argc,char ** argv);

int Init11Param_Main(int argc,char ** argv);
int New_Tapas_main(int,char **);
int GCPCtrl_main(int,char **);
int GCPVisib_main(int,char **);
int MakeMultipleXmlXifInfo_main(int argc,char ** argv);

int AddAffinity_main(int argc, char **argv);

int Sake_main(int argc, char ** argv);
int SateLib_main(int argc, char ** argv);

int DoAllDev_main(int argc,char ** argv);



#if (ELISE_X11)
    int SaisieAppuisInit_main(int argc,char ** argv);
    int SaisieAppuisPredic_main(int argc,char ** argv);
    int SaisieBasc_main(int argc,char ** argv);
    int SaisieCyl_main(int argc,char ** argv);
    int SaisieMasq_main(int argc,char ** argv);
    int SaisiePts_main(int argc,char ** argv);
    int SEL_main(int argc,char ** argv);
    int MICMACSaisieLiaisons_main(int argc,char ** argv);

    #ifdef ETA_POLYGON
        // Etalonnage polygone
        int Compens_main(int argc,char ** argv);
        int CatImSaisie_main(int argc,char ** argv);
        int CalibFinale_main(int argc,char ** argv);
        int CalibInit_main(int argc,char ** argv);
        int ConvertPolygone_main(int argc,char ** argv);
        int PointeInitPolyg_main(int argc,char ** argv);
        int RechCibleDRad_main(int argc,char ** argv);
        int RechCibleInit_main(int argc,char ** argv);
        int ScriptCalib_main(int argc,char ** argv);
    #endif

#endif

#if ELISE_QT
    int SaisieMasqQT_main(int argc,char ** argv);
    int SaisieAppuisInitQT_main(int argc,char ** argv);
    int SaisieAppuisPredicQT_main(int argc,char ** argv);
    int SaisieBoxQT_main(int argc,char ** argv);
    int SaisieBascQT_main(int argc,char ** argv);
    int SaisieCylQT_main(int argc,char ** argv);
#endif
  int ServiceGeoSud_TP2GCP_main(int argc, char **argv);
  int ServiceGeoSud_Ortho_main(int argc, char **argv);
  int ServiceGeoSud_GeoSud_main(int argc, char **argv);
  int ServiceGeoSud_Surf_main(int argc, char **argv);

int TopoSurf_main(int argc, char **argv);

int  CalcAutoCorrel_main(int argc,char ** argv);
int CPP_AppliMergeCloud(int argc,char ** argv);
int C3DC_main(int argc,char ** argv);
int MPI_main(int argc,char ** argv);
int MPI2Ply_main(int argc,char ** argv);
int MPI2Mnt_main(int argc,char ** argv);
int CCL_main(int argc,char ** argv);
int TDEpip_main(int argc, char **argv);
int Sat3D_main(int argc, char **argv);
int TiePHistoP_main(int argc, char **argv);

int TestNewOriImage_main(int argc,char ** argv);
int TestAllNewOriImage_main(int argc,char ** argv);
int PreparSift_Main(int argc,char ** argv);
int CheckOneHom_main(int argc,char ** argv);
int CheckAllHom_main(int argc,char ** argv);
int CheckOneTiff_main(int argc,char ** argv);
int CheckAllTiff_main(int argc,char ** argv);
int MakeOneXmlXifInfo_main(int argc,char ** argv);

int Masq3Dto2D_main(int argc,char ** argv);

int CPP_AppliMergeCloud(int argc,char ** argv);
int MMEnveloppe_Main(int argc,char ** argv);
int PlySphere_main(int argc,char ** argv);
int San2Ply_main(int argc,char ** argv);
int Export2Ply_main(int argc,char **argv);
int CASALL_main(int argc,char ** argv);
int MatisOri2MM_main(int argc,char ** argv);

extern int MMEnvStatute_main(int argc,char ** argv);



int VisuResiduHom(int argc,char ** argv);



void Paral_Tiff_Dev
    (
         const std::string & aDir,
         const std::vector<std::string> & aLFile,
         int                            aNbChan,
         bool                           Cons16B
    );


#endif // ElMemberTpl

#endif // _ELISE_GENERAL_ARG_MAIN_H




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant �  la mise en
correspondances d'images pour la reconstruction du relief.

Ce logiciel est régi par la licence CeCILL-B soumise au droit français et
respectant les principes de diffusion des logiciels libres. Vous pouvez
utiliser, modifier et/ou redistribuer ce programme sous les conditions
de la licence CeCILL-B telle que diffusée par le CEA, le CNRS et l'INRIA
sur le site "http://www.cecill.info".

En contrepartie de l'accessibilité au code source et des droits de copie,
de modification et de redistribution accordés par cette licence, il n'est
offert aux utilisateurs qu'une garantie limitée.  Pour les mêmes raisons,
seule une responsabilité restreinte pèse sur l'auteur du programme,  le
titulaire des droits patrimoniaux et les concédants successifs.

A cet égard  l'attention de l'utilisateur est attirée sur les risques
associés au chargement,  �  l'utilisation,  �  la modification et/ou au
développement et �  la reproduction du logiciel par l'utilisateur étant
donné sa spécificité de logiciel libre, qui peut le rendre complexe �
manipuler et qui le réserve donc �  des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités �  charger  et  tester  l'adéquation  du
logiciel �  leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement,
�  l'utiliser et l'exploiter dans les mêmes conditions de sécurité.

Le fait que vous puissiez accéder �  cet en-tête signifie que vous avez
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

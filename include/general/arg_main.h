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

void MemoArg(int,char**);
void ShowArgs();


class ElGramArgMain  // classe contenant la "grammaire" rudimenataire
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

extern bool Str2Bool(bool & aRes,const std::string & aStr);
extern bool Str2BoolForce(const std::string & aStr);

template <> inline std::istream & ElStdRead (std::istream &is, bool & aVal, const ElGramArgMain & G)
{
   std::string aStr ;
   is >> aStr;
   Str2Bool(aVal,aStr);
   return is;
}


template <class Type>  std::istream & operator >> (std::istream &is,ElSTDNS vector<Type> & vec);

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


class GenElArgMain
{
	public :
		virtual ~GenElArgMain()  {};
	    GenElArgMain(const char * Name,bool ISINIT ) ;
		virtual GenElArgMain * dup() const = 0;

		virtual void InitEAM(const ElSTDNS string &s,const ElGramArgMain &) const = 0;
		bool InitIfMatchEq(const ElSTDNS string &s,const ElGramArgMain & Gram) const;

		bool IsInit() const;
		const char *name() const;

                virtual void show(bool named) const =0;

          // Ensemble de patch pour rajouter des argument inutilise a la volee
		bool IsActif() const;

		static const char * ActifStr(bool);

    protected :
                static const std::string  theChaineInactif;
                static const std::string  theChaineActif;

		ElSTDNS string	_name;
		mutable bool	_is_init;
};


template <class Type> const char * str_type(Type *);


extern std::set<void *>  AllAddrEAM;

template <class Type> class ElArgMain : public GenElArgMain
{
	public :

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
                   const std::string & aCom = ""
             ) :
			GenElArgMain(Name,isInit),
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

	private :

		Type * 	    _adr;
                std::string  mCom;

};

bool EAMIsInit(void *);

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
                            const std::string aComment = ""
                     )
{
		return ElArgMain<Type>(v,Name,isInit,aComment);
}
template <class Type> ElArgMain<Type> 
                     EAMC
                     (
                            Type & v,
                            const std::string aComment 
                     )
{
                AllAddrEAM.insert( (void *) &v);
		return ElArgMain<Type>(v,"",false,aComment);
}



class LArgMain
{
	public :
		
		template <class Type> LArgMain & operator << (const ElArgMain<Type> & v)
		{
		        if (v.IsActif())
			   _larg.push_back(v.dup());
			return * this;
		}
		~LArgMain();


		INT  Init(int argc,char ** argv) const;
                void  InitIfMatchEq
                      (
                          std::vector<char *> *,  // Si !=0, empile les arg non consommes
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
                // Apparemment certain compilo
                // utilise la copie en temporaire;
		//      LArgMain(const LArgMain &); 
		// void operator = (const LArgMain &);
};





// Renvoie eventuellement la parti non consommee
std::vector<char *>   	ElInitArgMain
		(
			int argc,char ** argv,
			const LArgMain & ,
			const LArgMain & ,
                        bool  VerifInit=true,
                        bool  AccUnK=false,
			int   aNbArgGlobGlob = -1 // 
		);

void  	ElInitArgMain
		(
			const std::string &,
			const LArgMain & ,
			const LArgMain & 
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
        static bool ReadFormat(char  & aCom,std::string & aFormat,const std::string aFileOrLine,bool IsFile);

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

// uti_images
int Undist_main(int argc,char ** argv);
int Dequant_main(int argc,char ** argv);
int Devlop_main(int argc,char ** argv);
int ElDcraw_main(int argc,char ** argv);
int GenXML2Cpp_main(int argc,char ** argv);
int GrShade_main(int argc,char ** argv);
int MpDcraw_main(int argc,char ** argv);
int PastDevlop_main(int argc,char ** argv);
int Reduc2MM_main(int argc,char ** argv);
int ScaleIm_main(int argc,char ** argv);
int tiff_info_main(int argc,char ** argv);
int to8Bits_main(int argc,char ** argv);
int MPDtest_main(int argc,char ** argv);

// uti_phgram
int AperiCloud_main(int argc,char ** argv);
int Apero_main(int argc,char ** argv);
int Bascule_main(int argc,char ** argv);
int CmpCalib_main(int argc,char ** argv);
int Campari_main(int argc,char ** argv);
int GCPBascule_main(int argc,char ** argv);
int MakeGrid_main(int argc,char ** argv);
int Malt_main(int argc,char ** argv);
int Mascarpone_main(int argc,char ** argv);
int MergePly_main(int argc,char ** argv);
int MICMAC_main(int argc,char ** argv);
int Nuage2Ply_main(int argc,char ** argv);
int Pasta_main(int argc,char ** argv);
int Pastis_main(int argc,char ** argv);
int Porto_main(int argc,char ** argv);
int ReducHom_main(int argc,char ** argv);
int RepLocBascule_main(int argc,char ** argv);
int SBGlobBascule_main(int argc,char ** argv);
int Tapas_main(int argc,char ** argv);
int Tapioca_main(int argc,char ** argv);
int Tarama_main(int argc,char ** argv);
int Tawny_main(int argc,char ** argv);
int TestCam_main(int argc,char ** argv);
int ScaleNuage_main(int argc,char ** argv);
int  Gri2Bin_main(int argc,char ** argv);

int MMPyram_main(int argc,char ** argv);
int CreateEpip_main(int argc,char ** argv);
int AperoChImMM_main(int argc,char ** argv);
int MMInitialModel_main(int argc,char ** argv);
int CheckDependencies_main(int argc,char ** argv);
int NuageBascule_main(int argc,char ** argv);
int  cod_main(int argc,char ** argv);
int GCP_Txt2Xml_main(int argc,char ** argv);



#if (ELISE_X11)
int SaisieAppuisInit_main(int argc,char ** argv);
int SaisieAppuisPredic_main(int argc,char ** argv);
int SaisieBasc_main(int argc,char ** argv);
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


#endif // ElMemberTpl

#endif // _ELISE_GENERAL_ARG_MAIN_H




/*Footer-MicMac-eLiSe-25/06/2007

Ce logiciel est un programme informatique servant à la mise en
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
associés au chargement,  à l'utilisation,  à la modification et/ou au
développement et à la reproduction du logiciel par l'utilisateur étant 
donné sa spécificité de logiciel libre, qui peut le rendre complexe à 
manipuler et qui le réserve donc à des développeurs et des professionnels
avertis possédant  des  connaissances  informatiques approfondies.  Les
utilisateurs sont donc invités à charger  et  tester  l'adéquation  du
logiciel à leurs besoins dans des conditions permettant d'assurer la
sécurité de leurs systèmes et ou de leurs données et, plus généralement, 
à l'utiliser et l'exploiter dans les mêmes conditions de sécurité. 

Le fait que vous puissiez accéder à cet en-tête signifie que vous avez 
pris connaissance de la licence CeCILL-B, et que vous en avez accepté les
termes.
Footer-MicMac-eLiSe-25/06/2007*/

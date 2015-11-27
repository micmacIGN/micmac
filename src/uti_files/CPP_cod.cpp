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

#include "StdAfx.h"

#if (ELISE_windows)&(!ELISE_MinGW)
	inline int isblank( int c ){ return ( c==int(' ') || c==int('\t') ); }
#endif

Im1D_U_INT1 ImMajic()
{
	static Im1D_U_INT1 res(1);
	static bool First = true;
	if (First)
	{
		First = false;
		const char * majic = "krznp re9pjsuquk8peyk9mcnbwlmqopa6teyioptrenslohteapoiutegnm";
		res = Im1D_U_INT1( (int)strlen(majic) );

		for ( INT x=0; x<res.tx(); x++ )
			res.data()[x] = majic[x];
	}
	return res;
}

// Im1D_U_INT1 Majic = ImMajic();
string PostCode("dcd");

bool code_file(const char * name,bool coder,std::string * ResNewName=0)
{
    std::string aStrName(name);
    bool dcdpost = false;

    string NewName (name);

    if (IsPostfixed(name)) 
    {
        dcdpost = (StdPostfix(name)=="dcd");
    }
    else
    {
    }

    if (coder == dcdpost)
       return false;


    if (coder)
        NewName = name + string(".dcd");
    else
        NewName = StdPrefix(name); 

    if (ResNewName) * ResNewName = NewName;

    std::string aSauv = DirOfFile(aStrName) + "Dup_" + NameWithoutDir(aStrName) + ".dup";
    std::string aCp = "cp " + aStrName  + " " + aSauv;
    VoidSystem(aCp.c_str());



    string MV = string(SYS_MV)+ " \"" + name + string("\" \"") + NewName +string("\"");

    INT NbOctet = sizeofile(name);
    Elise_File_Im  F(name, Pt2di(NbOctet,1),GenIm::u_int1);
	Im1D_U_INT1 majic = ImMajic();
    ELISE_COPY
    (
           F.all_pts(),
           F.in()^majic.in()[FX%majic.tx()],
           F.out()
    );

    // cout << MV.c_str() << "\n";
    VoidSystem(MV.c_str());
    ELISE_fp::RmFile(aSauv);

    return true;
}

class FileCode : public ElActionParseDir
{

    public :
       void act(const ElResParseDir & res) 
       {
         if (res.is_dir())
            return;
         code_file(res.name(),_coder);
       }

        FileCode(bool coder) : _coder (coder) {}

    private :

      bool _coder;
};

int cod_main(int argc,char ** argv)
{
    string Name;
    INT decoder = 0;

    ElInitArgMain
    (
        argc,argv,
        LArgMain() 	<< EAM(Name) ,
        LArgMain() 	 << EAM(decoder,"dc",true)
    );	

    FileCode FC(!decoder);
    // code_file(Name.c_str());
    ElParseDir(strdup(Name.c_str()),FC);
    return 0;
}

void decoder_force(const char * aName,std::string & aNew)
{
    bool aOk = code_file(aName,false,&aNew);
    if (! aOk)
    {
        std::cout << "For name " << aName << "\n";
        ELISE_ASSERT(false,"cannot decode");
    }
}

void coder_force(const char * aName)
{
    bool aOk = code_file(aName,true);
    if (! aOk)
    {
        std::cout << "For name " << aName << "\n";
        ELISE_ASSERT(false,"cannot code");
    }
}


int vicod_main(int argc,char ** argv)
{
    string Name;
    std::string anEdit = "vi";

    ElInitArgMain
    (
        argc,argv,
        LArgMain() 	<< EAM(Name) ,
        LArgMain() 	 << EAM(anEdit,"editor",true)
    );	

    string aNewName;
    decoder_force(Name.c_str(),aNewName);

    std::string  aCom = anEdit + " " + aNewName;
    system_call(aCom.c_str());

    coder_force(aNewName.c_str());

    return 0;
}


//  =========================== GESTION DES ADRESSES MAIL =======================

class cOneEntryMail
{
     public : 
         bool  operator < (const cOneEntryMail & anE2) const
         {
               if (mBlackList != anE2.mBlackList)  return mBlackList;

               if (mAffil <anE2.mAffil) return true;
               if (mAffil >anE2.mAffil) return false;
               return mName < anE2.mName;
         }
         static cOneEntryMail * mLastEntry;

         cOneEntryMail(const std::string &,bool IsBlackL);
         std::string  mAdr;
         bool         mBlackList;
         std::string  mId; 
         std::string  mName;
         std::string  mAffil;
         bool         mOk;
};

cOneEntryMail*  cOneEntryMail::mLastEntry=0;

class cCmpOEM
{
    public :
       bool operator() (const cOneEntryMail * anE1,const cOneEntryMail * anE2)
       {
            return (*anE1) < (*anE2);
       }
};



class  cFoncCarBool
{
     public :
        cFoncCarBool  (bool (*aF)(int))
        {
            for (int aC=0 ; aC<256 ; aC++)
               mLut[aC] = aF(aC);
        }
        bool operator() (int aC) {return (aC>=0)&&(aC<256) && (mLut[aC]);}
      private :
        bool mLut[256];
};



bool FoncIsMailSep(int aC)
{
   
   return     isblank(aC)
           || (aC==',')
           || (aC==';')
           || (aC=='<')
           || (aC=='>')
           || (aC==':')
           || (aC=='\'')
           || (aC=='/')
           || (aC=='+')
           || (aC=='"')
           || (aC=='(')
           || (aC==')')
           || (aC=='=')
           || (aC=='?')
           || (aC=='[')
           || (aC==']')
           || (aC=='#')
           || (aC=='*')
           || (aC=='!')
/*
           || (aC=='{')
           || (aC=='}')
*/
           || (!isascii(aC))
           || (aC==10)
           || (aC==92)
          ;
}

cFoncCarBool IsMailSep(FoncIsMailSep);

bool FoncIsMailCar(int aC)
{
   return     isalnum(aC)
           || (aC=='.')
           || (aC=='@')
           || (aC=='-')
           || (aC=='_');
}
cFoncCarBool IsMailCar(FoncIsMailCar);






cOneEntryMail::cOneEntryMail
(
   const std::string & anAdr,
   bool IsBlackL
) :
   mAdr (anAdr),
   mBlackList (IsBlackL)
{
   for (const char * aC= mAdr.c_str() ; *aC ; aC++)
   {
         mId += isalpha(*aC) ? tolower(*aC) : *aC;
   }
   bool Ok = SplitIn2ArroundCar(mId,'@',mName,mAffil,false);
   if (!Ok)
   {
        std::cout << "For adr " << mAdr << "\n";
        ELISE_ASSERT(false,"cOneEntryMail cannot split");
   }
   else
   {
        mOk =   (mName!="") && (mAffil!="");
        if (!mOk)
        {
             // std::cout << "NOTOK " << mId << "::"<< mName << "::" << mAffil << "\n";
        }
   }

   // std::cout << IsBlackL << " Id=[" << mId << "] Name=["  << mName << "] Affil=[" << mAffil << "]\n";
}


typedef enum
{
   eModeGMTest,
   eModeGMCreate
}
eTypeGMFile;

class cGenerateMail
{
     public :
        cGenerateMail(int argc,char ** argv);
     private :

        void ParseFile(const std::string &aName,bool aTest);
        bool OkAdr(const std::string &) const;
        bool OkDestAdr(const std::string &) const;
        

        std::string mDir;
        cInterfChantierNameManipulateur * mICNM;
        const cInterfChantierNameManipulateur::tSet *mNameFile;

        std::map <std::string,cOneEntryMail *> mDicE;
        std::vector<cOneEntryMail *>           mVE;
        std::vector<std::string>               mDests;
        int mNbByF;
        std::string                            mOnlyFile;
};

bool cGenerateMail::OkAdr(const std::string & anAdr) const
{
   return OkDestAdr(anAdr);
}

bool cGenerateMail::OkDestAdr(const std::string & anAdr) const
{
   if (mDests.empty()) 
      return true;
  
   if (! IsPostfixed(anAdr)) return false;

// std::cout << "ppppppppp " << StdPostfix(anAdr) << "\n";

  return BoolFind(mDests,StdPostfix(anAdr));
}


void cGenerateMail::ParseFile(const std::string &aName,bool aTest)
{
    cOneEntryMail::mLastEntry=0;

    bool aBlackL = (NameWithoutDir(aName)=="Black-Liste.txt.dcd");
    if (mOnlyFile != "")
    {
         aBlackL = (NameWithoutDir(aName) != mOnlyFile);
    }

    if (aBlackL) 
    {
          std::cout <<   "     #######  BLACKLIST FILE ####\n";
    }
    std::string aNewName;

    decoder_force(aName.c_str(),aNewName);
    FILE * aFP = FopenNN(aNewName,"r","cGenerateMail::open");

    int aC;
    std::string anAdr;
    int aNbArr=0;
    while ((aC=fgetc(aFP)) != EOF)
    {
          if (aTest)
          {
               std::cout << char(aC) ;
          }
          bool aSep = IsMailSep(aC);
          bool aCar = IsMailCar(aC);

          if (aSep==aCar)
          {
               std::cout << "\n";
               std::cout << "in file " << aName << "\n";
               std::cout << " C= " << char(aC) << " I="  << int(aC) << "\n";
               fclose(aFP);
               coder_force(aNewName.c_str());
               ELISE_ASSERT(false,"Unexptecd char");
          }

          if (aSep)
          {
             if (aNbArr==1)
             {
                  cOneEntryMail  anEntr(anAdr,aBlackL);
                  if (! mDicE[anEntr.mId])  
                  {
                      mDicE[anEntr.mId] = new cOneEntryMail(anEntr);
                      if (
                            (anEntr.mOk)
                            && (aBlackL ||OkAdr(anEntr.mId) )
                         )
                      {
                         mVE.push_back(mDicE[anEntr.mId]);
                      }
                      else
                      {
                           std::cout << "ERR !! :: For File " << aName  << " For Name " << anAdr ;
                           if (cOneEntryMail::mLastEntry)
                              std::cout << " After " <<  cOneEntryMail::mLastEntry->mAdr;
                           else
                              std::cout << " (First of File)";
                           std::cout << "\n";
                      }
                      cOneEntryMail::mLastEntry = mDicE[anEntr.mId];
                  }
                  else
                  {
                      if (0)
                        std::cout << "Multiple " << anAdr << "\n";
                  }
                  if (anEntr.mBlackList)
                     mDicE[anEntr.mId]->mBlackList = true;
             }
             anAdr = "";
             aNbArr=0;
          }
          if (aCar)
          {
              anAdr += aC;
              if (aC=='@') 
                 aNbArr++;
          }
    }

    fclose(aFP);
    coder_force(aNewName.c_str());
}




cGenerateMail::cGenerateMail(int argc,char ** argv) :
    mDir (MMDir() + "Documentation/Mailing/"),
    mICNM (cInterfChantierNameManipulateur::BasicAlloc(mDir)),
    mNameFile (mICNM->Get(".*\\.dcd")),
    mOnlyFile ("")
{
    mNbByF=298;
    ElInitArgMain
    (
        argc,argv,
        LArgMain() ,
        LArgMain() << EAM(mNbByF,"NbByF",true)
                   << EAM(mOnlyFile,"SingleFile",true,"If specified, all but this one will considered as black-list files")
                   << EAM(mDests,"Dests",true,"Selected dest (for ex [fr] if only french)")
    );	


    for (int aKN=0 ; aKN<int(mNameFile->size()) ; aKN++)
    {
        std::string aName = mDir + (*mNameFile)[aKN];
        std::cout << "========================= begin Name File :: " << aName << "\n";
        ParseFile(aName,false);
    }

    cCmpOEM TheCmp;
    std::sort(mVE.begin(),mVE.end(),TheCmp);

    FILE * aFP =0;
    int aCptF =0; 
    int aCptInF =0; 




    // std::string aSep=";";
    std::string aSep="";

    for (int aK=0 ; aK<int(mVE.size()) ; aK++)
    {
         if (aFP==0)
         {
              aFP = FopenNN(mDir+"MailList_"+ToString(aCptF)+".txt","w","MailList::open");
              fprintf(aFP,"marc.pierrot-deseilligny@ensg.eu%s\n",aSep.c_str());

         }
         if (! mVE[aK]->mBlackList)
         {
             fprintf(aFP,"%s",mVE[aK]->mAdr.c_str());
             if ((aCptInF==mNbByF) || ((aK+1)==int(mVE.size())))
             {
                  fprintf(aFP,"\n");
                  fprintf(aFP,"marc.deseilligny@gmail.com\n");
                  fclose(aFP);
                  aFP =0;
                  aCptF++;
                  aCptInF=0;
             }
             else
             {
                  aCptInF++;
                  fprintf(aFP,"%s\n",aSep.c_str());
             }
         }
    }
}


int  genmail_main(int argc,char ** argv)
{
   MMD_InitArgcArgv(argc,argv);

   cGenerateMail anAppli(argc,argv);

   return 1;
}



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

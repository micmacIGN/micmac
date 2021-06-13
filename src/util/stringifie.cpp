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


/*******************************************************/
/*                                                     */
/*                      ::                             */
/*                                                     */
/*******************************************************/


const std::string Terminator = "//#_-=+{}@$##$##@";

const std::string TheDirXmlGen=std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR;


void Stringify
     (
         const std::string &aNameInput,
         const std::string &aNameOutput,
         const std::string &aNameString
     )
{
   int aNbLigneTot=0;
   {
       FILE * aFIn = FopenNN(aNameInput,"rb","Stringify");

       int aC;
       while ((aC=fgetc(aFIn)) != EOF)
       {
          if (aC=='\n')
             aNbLigneTot++;
       }
       ElFclose(aFIn);
   }


   FILE * aFIn = FopenNN(aNameInput,"rb","Stringify");
   FILE * aFOut = FopenNN(aNameOutput,"wb","Stringify");

    //fprintf(aFOut,"%s","#include \"general/all.h\"");
    //fprintf(aFOut,"%s","#include \"private/all.h\"");
    //fprintf(aFOut,"%s",""");

   fprintf(aFOut,"#include \"StdAfx.h\"\n");

   
   fprintf(aFOut,"const char * %s[%d] = {\n",aNameString.c_str(),aNbLigneTot+2);

   fprintf(aFOut,"\"");
   int aC;
   while ((aC=fgetc(aFIn)) != EOF)
   {
       if (aC=='\n')
         fprintf(aFOut,"\\n\",\n\"");
       else if (aC=='"')
         fprintf(aFOut,"\\\"");
       else if (aC=='\\')
         fprintf(aFOut,"\\\\");
       else
         fputc(aC,aFOut);
   }
   fprintf(aFOut,"\",\"%s\"};\n",Terminator.c_str());


   ElFclose(aFIn);
   ElFclose(aFOut);
}


void XML_StdStringify (const std::string &aNameInput)
{
   std::string aDir,aPref;
   SplitDirAndFile(aDir,aPref,aNameInput);
   aPref = StdPrefix(aPref);

   Stringify
   (
         aNameInput,
         std::string("CodeGenere")+ELISE_CAR_DIR+"File2String"+ELISE_CAR_DIR+"Str_"+aPref+std::string(".cpp"),
         "theNameVar_"+aPref
   );
}

void StdXMl2CppAndString(const std::string &aNameInput)
{
    XML_StdStringify(aNameInput);

   std::string aDir,aPref;
   SplitDirAndFile(aDir,aPref,aNameInput);
   aPref = StdPrefix(aPref);

    cElXMLTree aTreeSpec
               (
                  std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR
                 + aPref + std::string(".xml"));
    
    aTreeSpec.StdGenCppGlob
    (
         std::string("src")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR+ aPref + ".cpp",
         std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR + aPref + ".h",
         ""
    );

}

extern const char * theNameVar_ParamChantierPhotogram[];
extern const char * theNameVar_SuperposImage[];
extern const char * theNameVar_DefautChantierDescripteur[];

std::map<std::string,const char*> theDicoStringification;

void InitEntryStringifie()
{
   static bool Done=false;
   if (! Done)
   {
       Done=true;
       AddEntryStringifie
       (
          std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR+"ParamChantierPhotogram.xml",
          theNameVar_ParamChantierPhotogram,
          true
       );
       AddEntryStringifie
       (
          std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR+"SuperposImage.xml",
          theNameVar_SuperposImage,
          true
       );
       AddEntryStringifie
       (
          std::string("include")+ELISE_CAR_DIR+"XML_GEN"+ELISE_CAR_DIR+"DefautChantierDescripteur.xml",
          theNameVar_DefautChantierDescripteur,
          false
       );
   }
}

// extern const char ** theNameVar_ParamMICMAC;

void AddEntryStringifie(const std::string & aKey,const char ** aVal,bool formal)
{
   

    // std::cout << aKey << "\n";
    InitEntryStringifie();

   // Fuck, fuck and refuck to Visual-Microsoft et ses limites sur
   // les char static !!!!

   if (theDicoStringification[aKey]==0)
   {
       int aNbTot=0;
       for (const char ** aC= aVal ; std::string(*aC)!=Terminator ; aC++)
       {
           aNbTot += (int)strlen(*aC);
       }


       char * aBig = new char [aNbTot+1] ;
       char * aLast = aBig;

       for (const char ** aC= aVal ;  std::string(*aC)!=Terminator ; aC++)
       {
            for (const char * aP= *aC ; *aP ; aP++)
                *(aLast++) = *aP;
       }
       *aLast=0;


       theDicoStringification[aKey] = aBig;
   }
}

const char * GetEntryStringifie(const std::string & aKey)
{
    InitEntryStringifie();
    return theDicoStringification[aKey];
}

/*******************************************************/
/*                                                     */
/*                      cSTRVirtStream                 */
/*                                                     */
/*******************************************************/

class cSTRVirtStream : public cVirtStream
{
   public :
      int my_getc() 
      {
         return *(mCur++);
      }
      int my_eof() {return 0;}
      void my_ungetc(int aC)  
      {
           mCur--;
           ELISE_ASSERT(mCur>=mC0,"Too far in cSTRVirtStream::ungetc");
           ELISE_ASSERT(*mCur==aC,"Coherence in cSTRVirtStream::ungetc");
      }

      ~cSTRVirtStream() {}

      cSTRVirtStream(const char * aC0,const std::string & aName,bool isFilePredef,bool isFileSpec) :
         cVirtStream (aName,isFilePredef,isFileSpec),
         mCur        (aC0),
         mC0         (aC0)
       {
       }


        const char * Ending() {return mCur;}

  private :
     const char *  mCur;
     const char *  mC0;
};


/*******************************************************/
/*                                                     */
/*                      cIStStrVirstream               */
/*                                                     */
/*******************************************************/

class cIStStrVirstream : public cVirtStream
{
   public :
      int my_getc() 
      {
         char aC;
         mISS >> aC;
         return  aC;
      }
      int my_eof() {return CHAR_MIN;}
      void my_ungetc(int aC)  
      {
           mISS.putback(aC);
      }
      cIStStrVirstream(std::istringstream & anISS) :
         cVirtStream  ("cIStStrVirstream",false,false),
         mISS         (anISS)
      {
      }
  private :
     std::istringstream & mISS;
};



/*******************************************************/
/*                                                     */
/*                      cFILEVirtStream                */
/*                                                     */
/*******************************************************/

// Simplification MPD a la correction de bug GM2, il n'y a pas besoin de
// memoriser  dans le getc puisque c'est l'appelant qui se paye le boulot
// (ungetc passe la valeur a rebufferiser)
         
class cFILEVirtStream : public cVirtStream
{
   public :
      int my_getc() 
      {
         if (mBuf.empty())
            return  fgetc(mFP);

         int aRes = mBuf.back();
         mBuf.pop_back();
         return aRes;
      }

      int my_eof() {return EOF;}
     
       void my_ungetc(int aC)  
       {
// std::cout << "UNGETC " << aC << "\n";
          mBuf.push_back(aC);
       }

      ~cFILEVirtStream() {ElFclose(mFP);}
      cFILEVirtStream(FILE * aFP,const std::string & aName,bool isFilePredef,bool isFileSpec) :
          cVirtStream(aName,isFilePredef,isFileSpec),
          mFP (aFP)
       {
       }

       void fread(void *dest,int aNbOct)
       {
            int aGot = (int)::fread(dest,1,aNbOct,mFP);
            ELISE_ASSERT(aNbOct==aGot,"cFILEVirtStream::fread");
       }


/*
isFilePredef,bool isFileSpec) :
         cVirtStream (aName,isFilePredef,isFileSpec),

  Inutile ?
       virtual FILE* getFilePtr()
       {
               return mFP;
       }
*/

  private :
     FILE * mFP;
     std::vector<int> mBuf;
};




/*******************************************************/
/*                                                     */
/*                      cVirtStream                    */
/*                                                     */
/*******************************************************/
cVirtStream::~cVirtStream() {}

cVirtStream::cVirtStream(const std::string & aName,bool isPreDef,bool IsSpec) :
  mName       (aName),
  mIsPredef   (isPreDef),
  mIsSpec     (IsSpec)
{
}

bool cVirtStream::IsFilePredef() const  {return mIsPredef;}
bool cVirtStream::IsFileSpec() const    {return mIsSpec;}

const  std::string & cVirtStream::Name() 
{
   return mName;
}

void cVirtStream::fread(void *dest,int aNbOct)
{
   ELISE_ASSERT(false,"No cVirtStream::fread");
}
const char * cVirtStream::Ending()
{
   ELISE_ASSERT(false,"No cVirtStream::fread");
   return 0;
}

extern void d(const char * n);

cVirtStream *  cVirtStream::StdOpen(const std::string & aName)
{
   std::string aNameSeul,aDir;
   SplitDirAndFile(aDir,aNameSeul,aName);
    
    bool isFilePredef = (aDir== MMDir()+TheDirXmlGen) || (aDir==TheDirXmlGen);
    bool isFileSpec =  isFilePredef && (aNameSeul!="DefautChantierDescripteur.xml");



    FILE *aFP = ElFopen(aName.c_str(),"rb");
    if (aFP!=0)
    {
       d(aName.c_str());
       return  new cFILEVirtStream(aFP,aName,isFilePredef,isFileSpec);
    }


   const char *  aStr = GetEntryStringifie(aName);
    if (aStr != 0)
    {
       return new cSTRVirtStream(aStr,aName,isFilePredef,isFileSpec);
    }




   std::cout <<  "For required file [" << aName << "]\n";
   ELISE_ASSERT(false,"Cannot open");

   return 0;
}

cVirtStream  * cVirtStream::VStreamFromCharPtr(const char* aCharPtr)
{
   return new cSTRVirtStream(aCharPtr,"VStreamFromCharPtr",false,false);
}

cVirtStream * cVirtStream::VStreamFromIsStream(std::istringstream & anISS)
{
    return new cIStStrVirstream(anISS);
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

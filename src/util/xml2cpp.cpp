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

cGlobXmlGen::cGlobXmlGen() :
   mPrec (-1)
{
}


static std::vector<int> mVPrec;
void  XMLPushContext(const cGlobXmlGen & aGXml)
{
    if (aGXml.mPrec>=0)
       mVPrec.push_back(aGXml.mPrec);
}
void  XMLPopContext(const cGlobXmlGen & aGXml)
{ 
    if (aGXml.mPrec>=0)
       mVPrec.pop_back();
}
int CurPrec()
{
   if (mVPrec.empty()) return 18;
   return mVPrec.back();
}

/***********************************************************/
/*                                                         */
/*                    cElXMLTree                           */
/*                                                         */
/***********************************************************/

static std::map<std::string,std::string> TheMapMangl;

bool TagFilsIsClass(const std::string & aNameTag)
{
   return (aNameTag != "Verbatim") && (aNameTag!="Herit");
}


void cElXMLTree::GenCppGlob
     (
          const std::string & aNameFile,
          const std::string & aNameSpace
     )
{
    StdGenCppGlob
    (
          aNameFile + ".cpp",
          aNameFile + ".h",
          aNameSpace
    );
}

void  cElXMLTree::StdGenCppGlob
      (
          const std::string & aNameCpp,
          const std::string & aNameH,
          const std::string & aNameSpace
      )
{
   FILE * aFileCpp = ElFopen(aNameCpp.c_str(),"w");
   ELISE_ASSERT
   (
      aFileCpp!=0,
      "Cannot open .cpp file in cElXMLTree::GenCpp"
   );

   FILE * aFileH = ElFopen(aNameH.c_str(),"w");
   ELISE_ASSERT
   (
      aFileH!=0,
      "Cannot open .h file in cElXMLTree::GenCpp"
   );
   // fprintf(aFileH,"#include \"general/all.h\"\n");
   // fprintf(aFileH,"#include \"private/all.h\"\n\n\n");
   // fprintf(aFileCpp,"#include \"%s\"\n\n\n",NameWithoutDir(aNameH).c_str());


   
   std::list<cElXMLTree *>  aLFilsCpp = GetAll("GenCpp");

   for
   (
       std::list<cElXMLTree *>::iterator itF=aLFilsCpp.begin();
       itF != aLFilsCpp.end();
       itF++
   )
   {
        (*itF)->GenOneCppNameSpace(aFileCpp,aFileH,aNameSpace);
   }

   ElFclose(aFileH);
   ElFclose(aFileCpp);
}

void cElXMLTree::Verbatim
     (
           FILE * aFileCpp,
           FILE * aFileH
     )
{
   std::string aNameFile = ValAttr("File");
   FILE *  aFp1 = 0;
   FILE *  aFp2 = 0;
   if (aNameFile==".h")
      aFp1 = aFileH;
   else if (aNameFile==".cpp")
      aFp1 = aFileCpp;
   else if (aNameFile==".h.cpp")
   {
      aFp1 = aFileH;
      aFp2 = aFileCpp;
   }
   else
   {
      ELISE_ASSERT
      (
          false,
          "Bad File attr in Verbatim clause"
      );
   }
   if (aFp1)
       fprintf(aFp1,"%s\n",Contenu().c_str());
   if (aFp2)
       fprintf(aFp2,"%s\n",Contenu().c_str());
}

void cElXMLTree::GenOneCppNameSpace
     (
           FILE * aFileCpp,
           FILE * aFileH,
	   std::string aNameSpace
     )
{
   bool hasIfnDef;
   std::string aNameIfDef = StdValAttr("Difndef",hasIfnDef);
   if (hasIfnDef)
   {
        fprintf(aFileH,"#ifndef %s\n",aNameIfDef.c_str());
        fprintf(aFileH,"#define %s\n",aNameIfDef.c_str());
        // fprintf(aFileCpp,"#ifndef %s\n",aNameIfDef.c_str());
   }
   
  
   aNameSpace = ValAttr("NameSpace",aNameSpace);
   if (aNameSpace != "")
   {
      // fprintf(aFileH,"XXXXX nAmespace %s{\n\n",aNameSpace.c_str());
      // fprintf(aFileCpp,"XXXXX nAmespace %s{\n\n",aNameSpace.c_str());
   }
   std::list<std::string> aLTypeLoc;
   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if ((*itF)->ValAttr("Class","false")=="true")
      {
          (*itF)->GenCppClass(aFileCpp,aFileH,aLTypeLoc,0);
	  aLTypeLoc.push_back((*itF)->NameOfClass());
      }
      else if ((*itF)->mValTag == "enum")
      {
           (*itF)->GenEnum(aFileCpp,aFileH); 
	   aLTypeLoc.push_back((*itF)->ValAttr("Name"));
      }
      else if ((*itF)->mValTag == "Verbatim")
      {
            (*itF)->Verbatim(aFileCpp,aFileH);
      }
   }

   if (aNameSpace != "")
   {
       fprintf(aFileH,"};\n");
       fprintf(aFileCpp,"};\n");
   }
   if (hasIfnDef)
   {
        fprintf(aFileH,"#endif // %s\n",aNameIfDef.c_str());
        // fprintf(aFileCpp,"#endif // %s\n",aNameIfDef.c_str());
   }

}

void cElXMLTree::GenEnum
     (
           FILE * aFileCpp,
           FILE * aFileH
     )
{
    cMajickChek aMj;
    aMj.Add("enum");
    std::string aName = ValAttr("Name");
    fprintf
    (
        aFileH,
        "typedef enum\n"
	"{\n"
    );



   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
       if (itF!= mFils.begin())
          fprintf (aFileH, ",\n");
       std::string aValue = (*itF)->ValAttr("Value","");
       if (aValue=="")
       {
          fprintf (aFileH, "  %s", (*itF)->mValTag.c_str());
          aMj.Add((*itF)->mValTag);
       }
       else
       {
          fprintf (aFileH, "  %s = %s", (*itF)->mValTag.c_str(),aValue.c_str());
          aMj.Add((*itF)->mValTag);
          aMj.Add(aValue);
       }
   }
   aMj.Add(aName);

    fprintf
    (
        aFileH,
	"\n} %s;\n",
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"void xml_init(%s & aVal,cElXMLTree * aTree);\n",
	aName.c_str()
    );

    // A cause d'ambiguite avec les ToString<int> , on change le nom
    fprintf
    (
        aFileH,
	"std::string  eToString(const %s & aVal);\n\n",
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"%s  Str2%s(const std::string & aName);\n\n",
	aName.c_str(),
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"cElXMLTree * ToXMLTree(const std::string & aNameTag,const %s & anObj);\n\n",
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"void  BinaryDumpInFile(ELISE_fp &,const %s &);\n\n",
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"std::string  Mangling( %s *);\n\n",
	aName.c_str()
    );
    fprintf
    (
        aFileH,
	"void  BinaryUnDumpFromFile(%s &,ELISE_fp &);\n\n",
	aName.c_str()
    );




    fprintf
    (
        aFileCpp,
	"%s  Str2%s(const std::string & aName)\n"
	"{\n",
	aName.c_str(),
	aName.c_str()
    );

    for
    (
        std::list<cElXMLTree *>::iterator itF=mFils.begin();
        itF != mFils.end();
        itF++
    )
    {

          fprintf
          (
	      aFileCpp,
	      "   %sif (aName==\"%s\")\n"
	      "      return %s;\n",
              (itF==mFils.begin()) ? ""  : "else ",
	      (*itF)->mValTag.c_str(),
	      (*itF)->mValTag.c_str()
          );
    }
    fprintf
    (
        aFileCpp,
	"  else\n"
	"  {\n"
	"      cout << aName << \" is not a correct value for enum %s\\n\" ;\n"
	"      ELISE_ASSERT(false,\"XML enum value error\");\n"
	"  }\n"
	"  return (%s) 0;\n"
	"}\n",
	aName.c_str(),
	aName.c_str()
    );
    //
    // xml_init 
    fprintf
    (
        aFileCpp,
	"void xml_init(%s & aVal,cElXMLTree * aTree)\n"
	"{\n"
	"   aVal= Str2%s(aTree->Contenu());\n"
	"}\n",
	aName.c_str(),
	aName.c_str()
    );
    
    fprintf
    (
        aFileCpp,
	"std::string  eToString(const %s & anObj)\n"
	"{\n",
	aName.c_str()
    );

    for
    (
        std::list<cElXMLTree *>::iterator itF=mFils.begin();
        itF != mFils.end();
        itF++
    )
    {

          fprintf
          (
	      aFileCpp,
	      "   if (anObj==%s)\n"
	      "      return  \"%s\";\n",
	      (*itF)->mValTag.c_str(),
	      (*itF)->mValTag.c_str()
          );
    }
    fprintf
    (
        aFileCpp,
        " std::cout << \"Enum = %s\\n\";\n"
	"   ELISE_ASSERT(false,\"Bad Value in eToString for enum value \");\n"
	"   return \"\";\n"
	"}\n\n",
        aName.c_str()
    );



    // ToXMLTree 
    fprintf
    (
        aFileCpp,
	"cElXMLTree * ToXMLTree(const std::string & aNameTag,const %s & anObj)\n"
	"{\n",
	aName.c_str()
    );
    fprintf
    (
        aFileCpp,
	"      return  cElXMLTree::ValueNode(aNameTag,eToString(anObj));\n"
	"}\n\n"
    );


    fprintf
    (
         aFileCpp,
         "void  BinaryDumpInFile(ELISE_fp & aFp,const %s & anObj)\n"
         "{\n"
         "   BinaryDumpInFile(aFp,int(anObj));\n"
         "}\n\n",
	  aName.c_str()
    );

    fprintf
    (
         aFileCpp,
         "void  BinaryUnDumpFromFile(%s & anObj,ELISE_fp & aFp)\n"
         "{\n"
         "   int aIVal;\n"
         "   BinaryUnDumpFromFile(aIVal,aFp);\n"
         "   anObj=(%s) aIVal;\n"
         "}\n\n",
	  aName.c_str(),
	  aName.c_str()
    );
    fprintf
    (
        aFileCpp,
	"std::string  Mangling( %s *) {return \"%s\";};\n\n",
	aName.c_str(),
	aMj.ShortMajId().c_str()
    );
/*
*/

     TheMapMangl[aName] =  aMj.ShortMajId();

	// "void  BinaryUnDumpFromFile(%s &,ELISE_fp &);\n\n",
}




std::string  cElXMLTree::NameOfClass()
{
   bool aGotRef;
   const std::string & aRefType =  StdValAttr("RefType",aGotRef);
   if (aGotRef)
      return std::string("c")+ aRefType;

   int aH = Profondeur();

   std::string aRes = (aH == 1)                ?
                      ValAttr("Type")          :
                      std::string("c")+mValTag ;
   return aRes;
}

bool cElXMLTree::IsDeltaPrec()
{
   return ValAttr("DeltaPrec","0") != "0";
}
                      

std::string  cElXMLTree::NameImplemOfClass()
{
   std::string aRes = NameOfClass();
   const std::string & aPat = ValAttr("Nb");

    if (aPat=="1") return aRes;
    if (aPat=="?") return   std::string("cTplValGesInit< ")
                          + aRes
                          + std::string(" >");

    if ((aPat=="*")||(aPat=="+"))
    {
       std::string aContainer = ValAttr("Container","std::list");

        if (aContainer=="std::map")
        {
            std::string  aKey =  ValAttr("KeyType");
            aRes = aKey+ "," + aRes ;
        }

       return    aContainer
               + std::string("< ")
               + aRes
               + std::string(" >");
    }

    std::cout << "     pour Tag=" << mValTag << "\n";
    ELISE_ASSERT(false,"cElXMLTree::NameOfClass() : Attribut Nb illicite");
    return "";
}

void cElXMLTree::GenStdXmlCall
     (
        const std::string & aPrefix,
        const std::string & aNamePere,
        const std::string & aNameObj,
        FILE * aFileCpp
     )
{
    const std::string & aPat = ValAttr("Nb");
    ELISE_ASSERT(!IsDeltaPrec(),"Unexpected IsDeltaPrec()");
    if ((aPat=="1") || (aPat=="?"))
    {
        std::string NoDef = "MP$$@#//[]ubamshaioxPKh167c_";
	const std::string & aDef = ValAttr("Def",NoDef);
	std::string aStrDef = "";
        if (aDef!=NoDef)
        {
            aStrDef = ","+ValAttr("Type",NoDef)+"("+aDef+")";
            if (ValAttr("Type",NoDef) == "std::string")
            {
                aStrDef = std::string(",std::string(\"")+aDef+std::string("\")");
            }
        }
        fprintf (aFileCpp,"\n");
        fprintf
        (
            aFileCpp,
            "%sxml_init(%s.%s(),%s->Get(\"%s\",1)%s); //tototo \n",
            aPrefix.c_str(),
            aNameObj.c_str(),
            mValTag.c_str(),
            aNamePere.c_str(),
            mValTag.c_str(),
	    aStrDef.c_str()
        );
    }
    else if  ((aPat=="*") || (aPat=="+"))
    {
        //MAP
        std::string aContainer = ValAttr("Container","std::list");
        std::string Arg3 = "";
        if (aContainer=="std::map") 
        {
              Arg3 = ",\"" + ValAttr("KeyGetVal") +"\"";
        }

        fprintf (aFileCpp,"\n");
        fprintf
        (
            aFileCpp,
            "%sxml_init(%s.%s(),%s->GetAll(\"%s\",false,1)%s);\n",
            aPrefix.c_str(),
            aNameObj.c_str(),
            mValTag.c_str(),
            aNamePere.c_str(),
            mValTag.c_str(),
            Arg3.c_str()
        );
    }
    else
       ELISE_ASSERT(false,"Unexpected XML Nb Pattern");
}

bool cElXMLTree::HasFilsPorteeGlob(const std::string & aName)
{
    cElXMLTree * aFils = Get(aName); 
    if (! aFils) 
       return false;
     std::string aPortee = aFils->ValAttr("Portee","Locale");
     if (aPortee == "Locale")
        return false;
    ELISE_ASSERT(aPortee == "Globale","Bad Value for attribute Portee")
    return true;
}


void  cElXMLTree::ModifMangling(cMajickChek & aMj)
{
    aMj.Add(NameOfClass());
    for
    (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
    )
    {
       if (TagFilsIsClass((*itF)->mValTag))  // Cas speciaux Herit/ Verbatim ....
       {
           const std::string & aPat = (*itF)->ValAttr("Nb");
           aMj.Add(aPat);
           aMj.Add((*itF)->mValTag);
           aMj.Add((*itF)->NameImplemOfClass());
           aMj.Add((*itF)->NameOfClass());
           aMj.Add(TheMapMangl[(*itF)->NameOfClass()]);
       }
    }
}


void cElXMLTree::GenCppClass
     (
         FILE * aFileCpp,
	 FILE* aFileH,
	 const std::list<std::string> & aLTypeLoc,
	 int aProf
     )
{
   int aH = Profondeur();

   if (aH<=1) return;

   bool hasRefType = HasAttr("RefType");
   if (hasRefType) return;

   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if (TagFilsIsClass((*itF)->mValTag))
      {
         (*itF)->GenCppClass(aFileCpp,aFileH,aLTypeLoc,aProf+1);
      }

   }

   std::string aNOC = NameOfClass();
   cMajickChek aMj;
   ModifMangling(aMj);
   TheMapMangl[aNOC]=aMj.ShortMajId();
    

   //  Generation du .h 

   fprintf(aFileH,"class %s",aNOC.c_str());

   {
      int aCpt = 0;
      for
      (
          std::list<cElXMLTree *>::iterator itF=mFils.begin();
          itF != mFils.end();
          itF++
      )
      {
         if ((*itF)->mValTag=="Herit")
         {
             aCpt ++;
             if (aCpt==1)
                fprintf(aFileH," :\n        ");
             if (aCpt>=2)
                fprintf(aFileH,",\n        ");
              fprintf
              (
                      aFileH,
                      "%s %s",
                      (*itF)->ValAttr("portee","private").c_str(),
                      (*itF)->Contenu().c_str()
              );
         }
      }
   }

   fprintf(aFileH,"\n");
   fprintf(aFileH,"{\n");
   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if ((*itF)->mValTag=="Verbatim")
      {
          (*itF)->Verbatim(0,aFileH);
      }
   }
   fprintf(aFileH,"    public:\n");
   fprintf(aFileH,"        cGlobXmlGen mGXml;\n\n");
   fprintf
   (
       aFileH,
       "        friend void xml_init(%s & anObj,cElXMLTree * aTree);\n\n",
       aNOC.c_str()
   );


   std::list<cElXMLTree *> aListH;
   GenAccessor(true,this,0,aFileH,aListH,true);
   std::list<cElXMLTree *> aListCpp;
   GenAccessor(true,this,0,aFileCpp,aListCpp,false);

   fprintf(aFileH,"    private:\n");
   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if (TagFilsIsClass((*itF)->mValTag))
      {
          fprintf
          (
                aFileH,
                "        %s m%s;\n",
                (*itF)->NameImplemOfClass().c_str(),
                (*itF)->mValTag.c_str()
          );
          if ((*itF)->IsDeltaPrec())
          {
             fprintf
             (
                   aFileH,
                   "        %s mGlob%s;\n",
                   (*itF)->NameOfClass().c_str(),
                   (*itF)->mValTag.c_str()
             );
          }     
      }
   }



   fprintf
   (
       aFileH,
       "};\n"
       "cElXMLTree * ToXMLTree(const %s &);\n\n",
        aNOC.c_str()
     );

    fprintf
    (
        aFileH,
	"void  BinaryDumpInFile(ELISE_fp &,const %s &);\n\n",
	aNOC.c_str()
    );
    fprintf
    (
        aFileH,
	"void  BinaryUnDumpFromFile(%s &,ELISE_fp &);\n\n",
	aNOC.c_str()
    );
    fprintf
    (
        aFileH,
	"std::string  Mangling( %s *);\n\n",
	aNOC.c_str()
    );


   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if ((*itF)->mValTag=="Verbatim")
      {
          (*itF)->Verbatim(aFileCpp,0);
      }
   }

   //============================================================
   //  Generation de BinaryDumpInFile  / BinaryUnDumpFromFile
   //============================================================

            // ========== BinaryUnDumpFromFile ==========
    fprintf
    (
        aFileCpp,
	"void  BinaryUnDumpFromFile(%s & anObj,ELISE_fp & aFp)\n"
	"{\n ",
	aNOC.c_str()
    );
    for
    (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
    )
    {
       if (TagFilsIsClass((*itF)->mValTag))  // Cas speciaux Herit/ Verbatim ....
       {
           const std::string & aPat = (*itF)->ValAttr("Nb");
           if (aPat=="1")
           {
               fprintf(aFileCpp,"    BinaryUnDumpFromFile(anObj.%s(),aFp);\n",(*itF)->mValTag.c_str());
           }
           else if (aPat=="?")
           {
               fprintf(aFileCpp,"  { bool IsInit;\n");
               fprintf(aFileCpp,"       BinaryUnDumpFromFile(IsInit,aFp);\n");
               fprintf
               (
                     aFileCpp,
                     "        if (IsInit) {\n"
                     "             anObj.%s().SetInitForUnUmp();\n"
                     "             BinaryUnDumpFromFile(anObj.%s().ValForcedForUnUmp(),aFp);\n"
                     "        }\n"
                     "        else  anObj.%s().SetNoInit();\n"
                    ,(*itF)->mValTag.c_str()
                    ,(*itF)->mValTag.c_str()
                    ,(*itF)->mValTag.c_str()
               );
               fprintf(aFileCpp,"  } ;\n");
           }
           else if ((aPat=="*") || (aPat=="+"))
           {
               std::string aContainer = (*itF)->ValAttr("Container","std::list");
               bool IsOk = (aContainer=="std::list") || (aContainer=="std::vector");
               if (IsOk)
               {
                    fprintf(aFileCpp,"  { int aNb;\n");
                    fprintf(aFileCpp,"    BinaryUnDumpFromFile(aNb,aFp);\n");
                    fprintf
                    (
                           aFileCpp,
                           "        for(  int aK=0 ; aK<aNb ; aK++)\n"
                           "        {\n"
                           "             %s aVal;\n"
                           "              BinaryUnDumpFromFile(aVal,aFp);\n"
                           "              anObj.%s().push_back(aVal);\n"
                           "        }\n"
                           ,(*itF)->NameOfClass().c_str()
                           ,(*itF)->mValTag.c_str()
                     );
                     fprintf(aFileCpp,"  } ;\n");
               }
               else
               {
                     fprintf(aFileCpp,"    ELISE_ASSERT(false,\"No Support for this conainer in bin dump\");\n");
               }
           }
       }
       // std::cout << " XXX " << aNOC << " " << (*itF)->mValTag << " " << TagFilsIsClass((*itF)->mValTag) << "\n";
    }

    fprintf ( aFileCpp, "}\n\n");

            // ========== BinaryDumpInFile ==========

    fprintf
    (
        aFileCpp,
	"void  BinaryDumpInFile(ELISE_fp & aFp,const %s & anObj)\n"
	"{\n",
	aNOC.c_str()
    );
    for
    (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
    )
    {
       if (TagFilsIsClass((*itF)->mValTag))  // Cas speciaux Herit/ Verbatim ....
       {
           const std::string & aPat = (*itF)->ValAttr("Nb");
           if (aPat=="1")
           {
               fprintf(aFileCpp,"    BinaryDumpInFile(aFp,anObj.%s());\n",(*itF)->mValTag.c_str());
           }
           else if (aPat=="?")
           {
               fprintf(aFileCpp,"    BinaryDumpInFile(aFp,anObj.%s().IsInit());\n",(*itF)->mValTag.c_str());
               fprintf
               (
                     aFileCpp
                    ,"    if (anObj.%s().IsInit()) BinaryDumpInFile(aFp,anObj.%s().Val());\n"
                    ,(*itF)->mValTag.c_str()
                    ,(*itF)->mValTag.c_str()
               );
           }
           else if ((aPat=="*") || (aPat=="+"))
           {
               std::string aContainer = (*itF)->ValAttr("Container","std::list");
               bool IsOk = (aContainer=="std::list") || (aContainer=="std::vector");
               if (IsOk)
               {
                    fprintf(aFileCpp,"    BinaryDumpInFile(aFp,(int)anObj.%s().size());\n",(*itF)->mValTag.c_str());
                    fprintf
                    (
                           aFileCpp,
                           "    for(  %s::const_iterator iT=anObj.%s().begin();\n"
                           "         iT!=anObj.%s().end();\n"
                           "          iT++\n"
                           "    )\n"
                           "        BinaryDumpInFile(aFp,*iT);\n"
                           ,(*itF)->NameImplemOfClass().c_str()
                           ,(*itF)->mValTag.c_str()
                           ,(*itF)->mValTag.c_str()
                     );
               }
               else
               {
                     fprintf(aFileCpp,"    ELISE_ASSERT(false,\"No Support for this conainer in bin dump\");\n");
               }
           }
       }
       // std::cout << " XXX " << aNOC << " " << (*itF)->mValTag << " " << TagFilsIsClass((*itF)->mValTag) << "\n";
    }

    fprintf ( aFileCpp, "}\n\n");
   //  Generation de xml_write 

   

   fprintf
   (
         aFileCpp,
          "cElXMLTree * ToXMLTree(const %s & anObj)\n"
         "{\n"
         "  XMLPushContext(anObj.mGXml);\n"
         "  cElXMLTree * aRes = new cElXMLTree((cElXMLTree *)0,\"%s\",eXMLBranche);\n",
         aNOC.c_str(),
         mValTag.c_str()
   );
   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
       if (TagFilsIsClass((*itF)->mValTag))
       {
           const std::string & aPat = (*itF)->ValAttr("Nb");
           std::string  aPortee = ((*itF)->Profondeur()>1) ? "" : "::";
           if (BoolFind(aLTypeLoc,(*itF)->ValAttr("Type","")))
	      aPortee = "";
           std::string  anArgSup =  "";
           if ((*itF)->Profondeur() <= 1)
              anArgSup = std::string("std::string(\"")
                         +(*itF)->mValTag
                         +std::string("\"),");
                              
           bool hasPL2XT;
           std::string aPL2XT = (*itF)->StdValAttr("PL2XmlTree",hasPL2XT);
           if (hasPL2XT)
               aPortee = aPL2XT;

           std::string aStRetag = "->ReTagThis(\""+ (*itF)->mValTag + "\")";
                              
           if (aPat=="?")
           {
              fprintf
              (
                 aFileCpp,
                 "   if (anObj.%s().IsInit())\n"
                 "      aRes->AddFils(%sToXMLTree(%sanObj.%s().Val())%s);\n",
                 (*itF)->mValTag.c_str(),
                  aPortee.c_str(),
                 anArgSup.c_str(),
                 (*itF)->mValTag.c_str(),
		 aStRetag.c_str()
              );
           }
           else if ((aPat=="*") || (aPat=="+"))
           {
              // MAP
              std::string aContainer = (*itF)->ValAttr("Container","std::list");
              std::string GetValOfIter = (aContainer=="std::map") ? ".second": "";
              fprintf
              (
                 aFileCpp,
                 "  for\n"
                 "  ( "
                 "      %s::const_iterator it=anObj.%s().begin();\n"
                 "      it !=anObj.%s().end();\n"
                 "      it++\n"
                 "  ) \n"
                 "      aRes->AddFils(%sToXMLTree(%s(*it)%s)%s);\n",
                 (*itF)->NameImplemOfClass().c_str(),
    //XXXX
                 (*itF)->mValTag.c_str(),
                 (*itF)->mValTag.c_str(),
                 aPortee.c_str(),
                 anArgSup.c_str(),
                 GetValOfIter.c_str(),
		 aStRetag.c_str()
              );
           }
           else
           {
              fprintf
              (
                 aFileCpp,
                 "   aRes->AddFils(%sToXMLTree(%sanObj.%s())%s);\n",
                 aPortee.c_str(),
                 anArgSup.c_str(),
                 (*itF)->mValTag.c_str(),
		 aStRetag.c_str()
              );
           }
       }
   }



   fprintf
   ( 
        aFileCpp, 
        "  aRes->mGXml = anObj.mGXml;\n"
        "  XMLPopContext(anObj.mGXml);\n"
        "  return aRes;\n"
        "}\n\n"
   );

   //  Generation de xml_init 

   fprintf(aFileCpp,"void xml_init(%s & anObj,cElXMLTree * aTree)\n",aNOC.c_str());
   fprintf(aFileCpp,"{\n");
// Modif MPD, intervertion des deux ligne precedente, ne buggait pas immediatement (comme devrait !)
// mais la correction semble resoudre un bug differe dans la lecture des xml ????
   fprintf(aFileCpp,"   if (aTree==0) return;\n");
   fprintf(aFileCpp,"   anObj.mGXml = aTree->mGXml;\n");

   for
   (
       std::list<cElXMLTree *>::iterator itF=mFils.begin();
       itF != mFils.end();
       itF++
   )
   {
      if (TagFilsIsClass((*itF)->mValTag))
      {
        if ((*itF)->IsDeltaPrec())
        {
            fprintf
            (
                aFileCpp,
                " \n"
                "  //  CAS SPECIAL Delta Prec\n"
                "  {\n"
                "     std::list<cElXMLTree *> aLTr = aTree->GetAll(\"%s\");\n"
                "     std::list<cElXMLTree *>::iterator itLTr = aLTr.begin();\n"
                "     xml_init(anObj.mGlob%s,*itLTr);\n"
                "     // itLTr++;\n"
                /*"     bool isFirst=true;" */
                "     while (itLTr!=aLTr.end())\n"
                "     {\n"
                "        %s aVal= anObj.mGlob%s;\n",
                (*itF)->mValTag.c_str(),
                (*itF)->mValTag.c_str(),
                (*itF)->NameOfClass().c_str(),
                (*itF)->mValTag.c_str()
            );
            for
            (
                std::list<cElXMLTree *>::iterator itF2=(*itF)->mFils.begin();
                itF2 != (*itF)->mFils.end();
                itF2++
            )
            {
                 (*itF2)->GenStdXmlCall("        ","(*itLTr)","aVal",aFileCpp);
                 fprintf
                 (
                     aFileCpp,
                     "        if ((*itLTr)->HasFilsPorteeGlob(\"%s\"))\n"
                     "          anObj.mGlob%s.%s() = aVal.%s();\n",
                     (*itF2)->mValTag.c_str(),
                     (*itF)->mValTag.c_str(),
                     (*itF2)->mValTag.c_str(),
                     (*itF2)->mValTag.c_str()
                 );
            }

            fprintf
            (
                aFileCpp,
                "\n"
                "        anObj.m%s.push_back(aVal);\n"
                "        itLTr++;\n"
                /*"        isFirst=false;\n" */
                "     }\n"
                "  }\n",
                (*itF)->mValTag.c_str()
            );
        }
        else
        {
           (*itF)->GenStdXmlCall("   ","aTree","anObj",aFileCpp);
        }
      }
   }
   fprintf(aFileCpp,"}\n\n");

    fprintf
    (
        aFileCpp,
	"std::string  Mangling( %s *) {return \"%s\";};\n\n",
	aNOC.c_str(),
	aMj.ShortMajId().c_str()
    );

   if (aProf <= 1 )
   {
      for (int aK=0 ; aK<3  ; aK++)
      {
         fprintf(aFileH,"/******************************************************/\n");
      }
   }
}


void cElXMLTree::GenAccessor
     (
          bool         Recurs,
          cElXMLTree * anAnc,
          int aProf,
          FILE* aFile,
          std::list<cElXMLTree *> & aList,  // Empile les fils pour imbrication
          bool isH
     )
{
// std::cout  << "GA " << mValTag << "\n";

   const std::string & aPat = ValAttr("Nb");
   aList.push_back(this);
   
   if (Recurs && ((aProf==0) || (aPat=="1") || (aPat=="?")))
   {
      for
      (
          std::list<cElXMLTree *>::iterator itF=mFils.begin();
          itF != mFils.end();
          itF++
      )
      {
           if (TagFilsIsClass((*itF)->mValTag))
           {
              std::string aDefAccFils = "true";
              bool GenAccFils = ((*itF)->ValAttr("AccessorFils",aDefAccFils)=="true");
              if ( !  HasAttr("RefType"))
                  (*itF)->GenAccessor(GenAccFils,anAnc,aProf+1,aFile,aList,isH);
           }
      }
   }
   if (aProf==0) 
      return;

   fprintf(aFile,"\n");
   std::string aPortee =  isH ? "" : (anAnc->NameOfClass()+"::");
   for (int aK=0 ; aK<2; aK++) // aK : 1 Const - 0 Non Const
   {
        std::string aContsQual = (aK==0) ? "" : "const ";

 // if (aProf !=1) fprintf(aFileH," // %d ",aProf);
        if (isH) fprintf(aFile,"        ");
        fprintf
        (
                aFile,
                "%s%s & %s%s()%s",
                aContsQual.c_str(),
                NameImplemOfClass().c_str(),
                aPortee.c_str(),
                mValTag.c_str(),
                aContsQual.c_str()
         );
         
         if (isH)
         {
             fprintf(aFile,";\n");
         }
         else
         {
             fprintf(aFile,"\n");
             fprintf(aFile,"{\n"); 
             fprintf(aFile,"   return ");
             if (aProf ==1)
                 fprintf(aFile,"m%s",mValTag.c_str());
             else
             {
                std::list<cElXMLTree *>::iterator itT = aList.begin();
                itT++;
                int aK=0;
                int aL = (int) aList.size();
                while (itT != aList.end())
                {
                   if (aK!=0)  fprintf(aFile,".");
                   fprintf(aFile,"%s()",(*itT)->mValTag.c_str());
                   const std::string & aPatLoc = (*itT)->ValAttr("Nb");
                   if ((aK!= aL-2) && (aPatLoc == "?"))
                      fprintf(aFile,".Val()");
                   itT++;
                   aK++;
                }
             }
             fprintf(aFile,";\n");
             fprintf(aFile,"}\n\n");
         }
   }  
   aList.pop_back();
}

#define aSzBuf  4000
bool GenereErrorOnXmlInit=true;
bool GotErrorOnXmlInit=false;

static char aBuf[aSzBuf];

void xml_init(std::string    & aStr,cElXMLTree * aTree)
{
   if (aTree->IsVide() )
     aStr = "";
   else
      aStr = aTree->Contenu();
}

bool Str2Bool(bool & aRes,const std::string & aStr)
{
    if ((aStr == "false") || (aStr=="0"))
    {
       aRes = false;
       return true;
    }
    if ((aStr == "true") || (aStr=="1"))
    {
       aRes = true;
       return  true;
    }
    return false;
}

bool Str2BoolForce(const std::string & aStr)
{
    bool aRes;
    bool Ok = Str2Bool(aRes,aStr);
    if (!Ok)
    {
       std::cout << "For Str=" << aStr << "\n";
       ELISE_ASSERT(Ok,"Bad value for bool");
    }

    return aRes;
}

void xml_init(bool  & aVal,cElXMLTree * aTree)
{
    if (Str2Bool(aVal,aTree->Contenu()))  
       return;
/*
    const std::string & aStr = aTree->Contenu();
    if ((aStr == "false") || (aStr=="0"))
       aVal = false;
    else if ((aStr == "true") || (aStr=="1"))
       aVal = true;
    else
*/
    {
       std::cout << "VAL=" << aVal << " Tag " << aTree->ValTag() << " Cont " << aTree->Contenu() << "\n";
       ELISE_ASSERT(false,"Unexpected value for xml::bool");
    }
}

void FuckQTReadFloat()
{
     setlocale(LC_ALL,"C");
}

void xml_init(double         & aVal,cElXMLTree * aTree)
{
   FuckQTReadFloat();

   int aNb = sscanf(aTree->Contenu().c_str(),"%lf %s",&aVal,aBuf);
   if (aNb!=1)
   {
        FuckQTReadFloat();
        aNb = sscanf(aTree->Contenu().c_str(),"%lf %s",&aVal,aBuf);
   }
   if (aNb!=1)
   {
      GotErrorOnXmlInit = true;
      if (GenereErrorOnXmlInit)
      {
          std::cout << "TAG = "<< aTree->ValTag()
                <<  " Nb= " << aNb 
                 << " Contenu=[" << aTree->Contenu() << "]"
                <<"\n";
          ELISE_ASSERT(false,"Bad Nb Value in xml_init (double)");
      }
    }
    else
    {
       GotErrorOnXmlInit = false;
    }
}

void xml_init(int    & aVal,cElXMLTree * aTree)
{
   int aNb = sscanf(aTree->Contenu().c_str(),"%d %s",&aVal,aBuf);
   if (aNb!=1)
   {
      GotErrorOnXmlInit = true;
      if (GenereErrorOnXmlInit)
      {
           std::cout << "TAG = "<< aTree->ValTag()
                <<  " Nb= " << aNb 
                 << " Contenu=[" << aTree->Contenu() << "]"
                <<"\n";
          ELISE_ASSERT(aNb==1,"Bad Nb Value in xml_init (int)");
      }
   }
   else
   {
       GotErrorOnXmlInit = false;
   }
}
void xml_init(Box2dr & aVal,cElXMLTree * aTree)
{
   int aNb = sscanf ( aTree->Contenu().c_str(), "%lf %lf %lf %lf %s", &aVal._p0.x, &aVal._p0.y, &aVal._p1.x, &aVal._p1.y, aBuf);

   if (aNb!=4)
   {
       FuckQTReadFloat();
       aNb = sscanf ( aTree->Contenu().c_str(), "%lf %lf %lf %lf %s", &aVal._p0.x, &aVal._p0.y, &aVal._p1.x, &aVal._p1.y, aBuf);
   }
   ELISE_ASSERT(aNb==4,"Bad Nb Value in xml_init (double)");
   aVal = Box2dr(aVal._p0,aVal._p1);
}

void xml_init(Box2di & aVal,cElXMLTree * aTree)
{
   int aNb = sscanf
             (
                aTree->Contenu().c_str(),
                "%d %d %d %d %s",
                &aVal._p0.x, &aVal._p0.y,
                &aVal._p1.x, &aVal._p1.y,
                aBuf
             );
   aVal = Box2di(aVal._p0,aVal._p1);
   ELISE_ASSERT(aNb==4,"Bad Nb Value in xml_init (double)");
}



void xml_init(Pt3dr & aP,cElXMLTree * aTree)
{
   int aNb = sscanf(aTree->Contenu().c_str(),"%lf %lf %lf %s",&aP.x,&aP.y,&aP.z,aBuf);

   if (aNb!=3)
   {
       FuckQTReadFloat();
       aNb = sscanf(aTree->Contenu().c_str(),"%lf %lf %lf %s",&aP.x,&aP.y,&aP.z,aBuf);
   }
   if (aNb!=3)
   {
       std::cout << "CONTENU=" << aTree->Contenu() << "\n";
       ELISE_ASSERT(aNb==3,"Bad Nb Value in xml_init (double)");
   }
}


void xml_init(Pt2dr & aP,cElXMLTree * aTree)
{
   int aNb = sscanf(aTree->Contenu().c_str(),"%lf %lf %s",&aP.x,&aP.y,aBuf);

   if (aNb!=2)
   {
       FuckQTReadFloat();
       aNb = sscanf(aTree->Contenu().c_str(),"%lf %lf %s",&aP.x,&aP.y,aBuf);
   }

   if (aNb!=2)
   {
      GotErrorOnXmlInit = true;
      if (GenereErrorOnXmlInit)
      {
          std::cout << "xml_init(Pt2dr..),"
                << " TAG=" <<  aTree->ValTag()
                << " Arg=" <<  aTree->Contenu()
		<< "\n";
          ELISE_ASSERT(aNb==2,"Bad Nb Value in xml_init (double)");
      }
   }
   else
   {
      GotErrorOnXmlInit = false;
   }
}

void xml_init(Pt2di & aP,cElXMLTree * aTree)
{
   int aNb = sscanf(aTree->Contenu().c_str(),"%d %d %s",&aP.x,&aP.y,aBuf);
   if (aNb!=2)
   {
       GotErrorOnXmlInit = true;
       if (GenereErrorOnXmlInit)
       {
          std::cout << "CONTENU = " << aTree->Contenu().c_str() << "\n";
          ELISE_ASSERT(false,"Bad Nb Value in xml_init (double)");
       }
   }
   else
   {
      GotErrorOnXmlInit = false;
   }
}

void xml_init(Pt3di & aP,cElXMLTree * aTree)
{
   int aNb = sscanf(aTree->Contenu().c_str(),"%d %d %d %s",&aP.x,&aP.y,&aP.z,aBuf);
   if (aNb!=3)
   {
       std::cout << "CONTENU = " << aTree->Contenu().c_str() << "\n";
       ELISE_ASSERT(false,"Bad Nb Value in xml_init (double)");
   }
}

void xml_init(cCpleString & aCple,cElXMLTree * aTree)
{
   //char aR1[200],aR2[200];
   char * aR1 = aBuf;
   char * aR2 = aBuf + aSzBuf/2;
   int aNb = sscanf(aTree->Contenu().c_str(),"%s %s %s",aR1,aR2,aBuf);
   if (aNb!=2)
   {
       std::cout << "CONTENU = " << aTree->Contenu().c_str() << "\n";
       ELISE_ASSERT(false,"Bad Nb Value in xml_init (cCpleString)");
   }
   aCple = cCpleString(aR1,aR2);
}

cMonomXY::cMonomXY() 
{
   *this = cMonomXY(0,0,0);
}

cMonomXY::cMonomXY(double aCoeff,int aDX,int aDY) :
    mCoeff (aCoeff),
    mDegX  (aDX),
    mDegY  (aDY)
{
}

void xml_init(cMonomXY & aPol,cElXMLTree * aTree)
{
   int aDX,aDY;
   double aCoeff;

   int aNb = sscanf(aTree->Contenu().c_str(),"%lf %d %d",&aCoeff,&aDX,&aDY);

   if (aNb!=3)
   {
       std::cout << "CONTENU = " << aTree->Contenu().c_str() << "\n";
       ELISE_ASSERT(false,"Bad Nb Value in xml_init (cMonomXY)");
   }
   aPol = cMonomXY(aCoeff,aDX,aDY);
}


void  xml_init(std::vector<double> & aV,cElXMLTree * aTree)
{
   FuckQTReadFloat();
   ElArgMain<std::vector<double> > anArg(aV,"toto",true);
   anArg.InitEAM(aTree->Contenu(),ElGramArgMain::StdGram);
}


void  xml_init(std::vector<int> & aV,cElXMLTree * aTree)
{
   ElArgMain<std::vector<int> > anArg(aV,"toto",true);
   anArg.InitEAM(aTree->Contenu(),ElGramArgMain::StdGram);
}


void  xml_init(std::vector<std::string> & aV,cElXMLTree * aTree)
{
   ElArgMain<std::vector<std::string> > anArg(aV,"toto",true);
   anArg.InitEAM(aTree->Contenu(),ElGramArgMain::StdGram);
}



void xml_init(cElRegex_Ptr & aPtrReg ,cElXMLTree * aTree)
{
   aPtrReg = new cElRegex(aTree->Contenu(),30);
   if (aPtrReg==0)
   {
      std::cout << "REGEX=[" << aTree->Contenu() << "]\n";
      ELISE_ASSERT(false,"Cannot Compile Regular Expression");
   }
}


void Debug2XmlT(int aLine)
{
    std::cout << " Debug2XmlT " << aLine << "\n";
}



#define XML_PRECISION(OS) OS.precision(CurPrec());

#define DEBUG_2XMLT  
//Debug2XmlT(__LINE__);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<double> & anObj)
{
   DEBUG_2XMLT
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << anObj;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<int> & anObj)
{
   DEBUG_2XMLT
   std::ostringstream anOS;
   // XML_PRECISION(anOS);
   anOS << anObj;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}


cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::vector<std::string> & anObj)
{
   DEBUG_2XMLT
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << anObj;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}




cElXMLTree * ToXMLTree(const std::string & aNameTag,const bool   &      anObj)
{
   DEBUG_2XMLT
   return  cElXMLTree::ValueNode(aNameTag,anObj ?"true":"false");
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const int   &      anObj)
{
   DEBUG_2XMLT
   sprintf(aBuf,"%d",anObj);
   return  cElXMLTree::ValueNode(aNameTag,aBuf);
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const double   &      anObj)
{
   DEBUG_2XMLT
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << anObj;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
 
   // sprintf(aBuf,"%e",anObj);
   ///  return  cElXMLTree::ValueNode(aNameTag,aBuf);
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const std::string   &      aStr)
{
   DEBUG_2XMLT
   return  cElXMLTree::ValueNode(aNameTag,aStr);
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const Box2dr   &      anObj)
{
   DEBUG_2XMLT
   // sprintf ( aBuf, "%lf %lf %lf %lf", anObj._p0.x,anObj._p0.y, anObj._p1.x,anObj._p1.y);
   // return  cElXMLTree::ValueNode(aNameTag,aBuf);
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << anObj._p0.x << " " << anObj._p0.y <<  " " <<  anObj._p1.x << " " << anObj._p1.y ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}


cElXMLTree * ToXMLTree(const std::string & aNameTag,const Box2di   &      anObj)
{
   DEBUG_2XMLT
   // sprintf ( aBuf, "%lf %lf %lf %lf", anObj._p0.x,anObj._p0.y, anObj._p1.x,anObj._p1.y);
   // return  cElXMLTree::ValueNode(aNameTag,aBuf);
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << anObj._p0.x << " " << anObj._p0.y <<  " " <<  anObj._p1.x << " " << anObj._p1.y ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}





cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2dr & aP)
{
   DEBUG_2XMLT
   // sprintf (aBuf,"%lf %lf",aP.x,aP.y);
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << aP.x << " " << aP.y ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
   // return  cElXMLTree::ValueNode(aNameTag,aBuf);
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2di & aP)
{
   DEBUG_2XMLT
   sprintf (aBuf,"%d %d",aP.x,aP.y);
   return  cElXMLTree::ValueNode(aNameTag,aBuf);
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt3di & aP)
{
   DEBUG_2XMLT
   sprintf (aBuf,"%d %d %d",aP.x,aP.y,aP.z);
   return  cElXMLTree::ValueNode(aNameTag,aBuf);
}



cElXMLTree * ToXMLTree(const std::string & aNameTag,const cCpleString & aCpl)
{
     std::ostringstream anOS;
     anOS << aCpl.N1() << " " << aCpl.N2() ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const cMonomXY & aMon)
{
    std::ostringstream anOS;
    anOS << aMon.mCoeff <<  " " << aMon.mDegX << " " << aMon.mDegY ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}


cElXMLTree * ToXMLTree(const Pt3dr & aP)
{
    ELISE_ASSERT(false,"ToXMLTree(const Pt3dr & aP)");
    return 0;
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt3dr & aP)
{
   DEBUG_2XMLT
   std::ostringstream anOS;
   XML_PRECISION(anOS);
   anOS << aP.x << " " << aP.y  << " " << aP.z ;
   return  cElXMLTree::ValueNode(aNameTag,anOS.str());
}

cElXMLTree * ToXMLTree(const std::string & aNameTag,const  cElRegex_Ptr & aPtr)
{
   DEBUG_2XMLT
   return  cElXMLTree::ValueNode(aNameTag,aPtr->NameExpr());
}

#define NEW_IM_XML true

extern void PutDataInXMLString(std::ostringstream & anOs,const void * aData,int aNbOct);
const void * GetDataInXMLString(std::istringstream & aStream,int aNbExpOct);



template <class T1,class T2>
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<T1,T2> & anIm)
{
   DEBUG_2XMLT


   std::ostringstream anOs;

   anOs << anIm.tx() << " ";
   anOs << anIm.ty() << "\n";

   int aNbOctet = anIm.tx()*anIm.ty()* sizeof(T1);

   if (NEW_IM_XML)
   {
      PutDataInXMLString(anOs,anIm.data_lin(),aNbOctet);
   }
   else
   {
      cConvertBaseXXX aCvtr = cConvertBaseXXX::StdBase64();
      aCvtr.PutNC(anIm.data_lin(),aNbOctet,anOs);

      aCvtr.Close(anOs);
   }


   cElXMLTree * aRes =  cElXMLTree::ValueNode(aNameTag,anOs.str());
   return aRes;
}


template <class T1,class T2> void xml_init( Im2D<T1,T2>  & anIm,cElXMLTree * aTree)
{
    std::istringstream anIs(aTree->Contenu());
    int aTx,aTy;
    anIs >> aTx;
    anIs >> aTy;
    int aNbOctets = aTx*aTy* sizeof(T1);

    const void * aData = GetDataInXMLString(anIs,aNbOctets);
    if (aData)
    {
        anIm =  Im2D<T1,T2>(aTx,aTy);
        memcpy(anIm.data_lin(),aData,aNbOctets);
    }
    else
    {
       anIm =  Im2D<T1,T2>(aTx,aTy);
       cConvertBaseXXX aCvtr = cConvertBaseXXX::StdBase64();
       aCvtr.GetNC(anIm.data_lin(),aNbOctets ,anIs);
    }
}


template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<REAL4,REAL8> & anIm);
template void xml_init(Im2D<REAL4,REAL8>  & anIm,cElXMLTree * aTree);

template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<REAL8,REAL8> & anIm);
template void xml_init(Im2D<REAL8,REAL8>  & anIm,cElXMLTree * aTree);

template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<U_INT1,INT> & anIm);
template void xml_init(Im2D<U_INT1,INT>  & anIm,cElXMLTree * aTree);

template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<U_INT2,INT> & anIm);
template void xml_init(Im2D<U_INT2,INT>  & anIm,cElXMLTree * aTree);

template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<INT1,INT> & anIm);
template void xml_init(Im2D<INT1,INT>  & anIm,cElXMLTree * aTree);


template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<INT2,INT> & anIm);
template void xml_init(Im2D<INT2,INT>  & anIm,cElXMLTree * aTree);

template cElXMLTree * ToXMLTree(const std::string & aNameTag,const Im2D<INT4,INT> & anIm); // MMVII
template void xml_init(Im2D<INT4,INT>  & anIm,cElXMLTree * aTree); // MMVII
//       TypeSubst

template <class Type> TypeSubst<Type>::TypeSubst() :
       mIsInit (false)
{
}

template <class Type> TypeSubst<Type>::TypeSubst(const Type & aVal) :
   mVal        (aVal),
   mIsInit     (true)
{
}

template <class Type> const  Type  & TypeSubst<Type>::Val() const
{
    if (! mIsInit)
    {
        std::cout << "For <" << mStrTag << ">=" << mStrInit << "\n";
        ELISE_ASSERT(false,"string with # has not been substituted");
    }
    return mVal;
}

template <class Type> void TypeSubst<Type>::SetStr(cElXMLTree * aTree)
{
    mStrInit = aTree->Contenu();
    mStrTag = aTree->ValTag();
    TenteInit();
}


template <class Type> bool  TypeSubst<Type>::Subst
                            (
                                bool AMMNoArg,  
                                const std::vector<std::string> & aVParam
                            )
{
   bool aRes = TransFormArgKey(mStrInit,AMMNoArg,aVParam);
   TenteInit();
   return aRes;
}

template <class Type> void TypeSubst<Type>::TenteInit()
{
    if ( mStrInit.find('#')==std::string::npos)
    {
       cElXMLTree * aTr = cElXMLTree::ValueNode(mStrTag,mStrInit);
       xml_init(mVal,aTr);
       delete aTr;
       mIsInit = true;
    }
    else
       mIsInit = false;
}


template class TypeSubst<bool>;

template class TypeSubst<int>;
template class TypeSubst<double>;
template class TypeSubst<Pt2di>;
template class TypeSubst<Pt2dr>;

void xml_init(IntSubst & anIS,cElXMLTree * aTree) { anIS.SetStr(aTree); }
void xml_init(DoubleSubst & anIS,cElXMLTree * aTree) { anIS.SetStr(aTree); }
void xml_init(Pt2diSubst & anIS,cElXMLTree * aTree) { anIS.SetStr(aTree); }
void xml_init(Pt2drSubst & anIS,cElXMLTree * aTree) { anIS.SetStr(aTree); }
void xml_init(BoolSubst & anIS,cElXMLTree * aTree) { anIS.SetStr(aTree); }



cElXMLTree * ToXMLTree(const std::string & aNameTag,const BoolSubst   &      anObj)   { return ToXMLTree(aNameTag,anObj.Val()); }
cElXMLTree * ToXMLTree(const std::string & aNameTag,const IntSubst   &      anObj)    { return ToXMLTree(aNameTag,anObj.Val()); }
cElXMLTree * ToXMLTree(const std::string & aNameTag,const DoubleSubst   &      anObj) { return ToXMLTree(aNameTag,anObj.Val()); }
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2diSubst   &      anObj)  { return ToXMLTree(aNameTag,anObj.Val()); }
cElXMLTree * ToXMLTree(const std::string & aNameTag,const Pt2drSubst   &      anObj)  { return ToXMLTree(aNameTag,anObj.Val()); }



XmlXml::XmlXml()
{
     mTree = cElXMLTree::ValueNode("XmlXml","");
}

void xml_init(XmlXml    & aXX,cElXMLTree * aTree)
{
   aXX.mTree = aTree->Clone();
}

cElXMLTree * cElXMLTree::Clone()
{
    mKind = eXMLClone;
    cElXMLTree * aRes = new  cElXMLTree(*this);
    aRes->mKind = eXMLClone;
    return aRes;
}


cElXMLTree * ToXMLTree(const std::string & aNameTag,const XmlXml &      anObj)
{
   return anObj.mTree->ReTagThis(aNameTag);
}

void BinaryDumpInFile(ELISE_fp &,const XmlXml &)
{
   ELISE_ASSERT(false,"No BinaryDumpInFilecfor XmlXml");
}

void BinaryUnDumpFromFile(XmlXml &,ELISE_fp &)
{
   ELISE_ASSERT(false,"No BinaryUnDumpFromFile XmlXml");
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

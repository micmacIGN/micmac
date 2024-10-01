#include "MMVII_PCSens.h"
#include "MMVII_MMV1Compat.h"
#include "MMVII_DeclareCste.h"
#include "MMVII_BundleAdj.h"
#include "MMVII_2Include_Serial_Tpl.h"
#include "MMVII_util_tpl.h"



/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{

/*   GCP : NXYZ  
 *   Mes32 : SijXYZS
 *   ORGI :
 *       IES
 *       NXYZ
 *       EEXY
 *       EEXY
 *  cAppli_StackDep
 *      SSS
 *  
 *  ISFSF
 *
 *
 *     case 'F' : aLNum.push_back(GetV<tREAL8>(iss,mNameFile,mNbLineRead));   initF++; break;
       case 'E' : aLInt.push_back(GetV<int>(iss,mNameFile,mNbLineRead));      initI++; break;
       case 'X' : aXYZ.x() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initXYZ++; break;
       case 'Y' : aXYZ.y() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initXYZ++;  break;
       case 'Z' : aXYZ.z() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initXYZ++; break;

       case 'W' : aWPK.x() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initWPK++; break;
       case 'P' : aWPK.y() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initWPK++; break;
       case 'K' : aWPK.z() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);         initWPK++; break;

        case 'i' : aij.x() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);          initij++;  break;
        case 'j' : aij.y() = GetV<tREAL8>(iss,mNameFile,mNbLineRead);          initij++;  break;

        case 'N' : aNamePt = GetV<std::string>(iss,mNameFile,mNbLineRead);     initPt++; break;
        case 'I' : aNameIm = GetV<std::string>(iss,mNameFile,mNbLineRead);     initIm++; break;
        case 'S' : aLString.push_back(GetV<std::string>(iss,mNameFile,mNbLineRead)); initString++; break;

 */




/// Represent the type of the field that will be stored
enum class eTypeField
{
      eInt,
      eFloat,
      eString,
      eUnused
};


class  cNewReadFilesStruct
{
     public :
         cNewReadFilesStruct(const std::string & aFormat,const std::string &  aSpecifFormat);
	 
	 void ReadFile(const std::string & aNameFile,int aL0,int aLL,int aCom);


	 /*
	 template <class Type> const std::vector<Type> &  GetVect(const std::string & aNameField)
	 {
             MMVII_INTERNAL_ASSERT_tiny(
	 }
	 */
     private :

	 template <class Type> void AddVal
		                    (
				        const std::string & aNameField, 
				        const std::string & aNameValue, 
				        std::map<std::string,std::vector<Type> >& aMap
                                    )
	 {
		 if (mDebug) 
                     StdOut() <<  "   " << aNameField << " = [" << aNameValue  << "] , '("<<  cStrIO<Type>::msNameType << ")\n";
		 aMap[aNameField].push_back(cStrIO<Type>::FromStr(aNameValue));
	 }
	 

	 /// Check that the specif and the format are coherent
	 void Check(std::map<std::string,size_t> & aMap1,std::map<std::string,size_t> & aMap2,bool RefIsI1);

	 ///  return the type of a given name
	 eTypeField  TypeOfName(const std::string &);

	 /// Parse a format specif to split in token and count the occurence of each token
         static void  ParseFormat(const std::string & aFormat,std::map<std::string,size_t>  & aMap,std::vector<std::string> & );

         std::string                     mFormat;      ///< Format used to parse the file 
	 bool mDebug;

	 std::map<std::string,size_t>    mCptFields;   ///< count the occurence of each token in the format
	 std::vector<std::string>        mNameFields;  ///< list of token
	 std::vector<eTypeField>         mTypes;       ///< list of type of token
						       //
	 std::string                     mNameFile;


	 std::map<std::string,std::vector<int> >          mMapInt;
	 std::map<std::string,std::vector<tREAL8> >       mMapFloat;
	 std::map<std::string,std::vector<std::string> >  mMapString;

};


void cNewReadFilesStruct::Check(std::map<std::string,size_t> & aMap1,std::map<std::string,size_t> & aMap2,bool RefIsI1)
{
     for (const auto & [aStr1,aCpt1] :  aMap1)
     {
	 if (TypeOfName(aStr1) ==  eTypeField::eUnused)
	 {
            // There is no sens to have skip in specif
            if (RefIsI1)
            {
                  MMVII_INTERNAL_ERROR("Skip in specif !!");
            }
	    else
	    {
		    // Everyrhing OK we can have as many skipped as wanted
	    }
	 }
	 else
	 {
              size_t aCpt2 = aMap2[aStr1];

	      if (aCpt1 != aCpt2)
	      {
                  MMVII_UserError(eTyUEr::eUnClassedError,"dif bewteen specif & format for "+aStr1);
	      }
	 }
     }
}

eTypeField cNewReadFilesStruct::TypeOfName(const std::string & aName)
{
     if (aName== "Sk")
        return eTypeField::eUnused;

     std::string aBeginF = "XYZWPKFS";
     std::string aBeginS = "NI";
     std::string aBeginI = "E";

     if (contains(aBeginF,aName.at(0))  )
        return eTypeField::eFloat;

     if (contains(aBeginS,aName.at(0))  )
        return eTypeField::eString;

     if (contains(aBeginI,aName.at(0))  )
        return eTypeField::eInt;

     MMVII_INTERNAL_ERROR("Cannot compile type of : " + aName);
     return eTypeField::eUnused;
}


void  cNewReadFilesStruct::ParseFormat(const std::string & aFormat,std::map<std::string,size_t>  & aMap,std::vector<std::string> & aVec)
{
    const char * aC = aFormat.c_str();

    while (*aC)
    {
        while (isspace(*aC) || (*aC==',') )  aC++;
        const char * aC0 = aC;
        if (isupper(*aC))  
        {
           aC++;
           while ( (*aC) && (!isupper(*aC)) && (isalnum(*aC))  )
               aC++;
        }
	else
	{
             MMVII_UserError(eTyUEr::eUnClassedError,"Format is not reconized  :" +std::string(aC));
	}
	std::string aNameField(aC0,size_t(aC-aC0));

        // StdOut() << "========== [" << aNameField << "]\n";
	aMap[aNameField] ++;
	aVec.push_back(aNameField);
    }
}

cNewReadFilesStruct::cNewReadFilesStruct(const std::string & aFormat,const std::string & aSpecifFormat) :
    mFormat  (aFormat),
    mDebug   (false)
{
   ParseFormat(mFormat,mCptFields,mNameFields);

   std::map<std::string,size_t>    aCptSpec;;
   std::vector<std::string>        aNameSpec;
   ParseFormat(aSpecifFormat,aCptSpec,aNameSpec);

   for (const auto & aStr: mNameFields)
        mTypes.push_back(TypeOfName(aStr));

   Check(mCptFields,aCptSpec,false);
   Check(aCptSpec,mCptFields,true);
}

void cNewReadFilesStruct::ReadFile(const std::string & aNameFile,int aL0,int aNumLastL ,int aComment)
{
    if (aNumLastL<=0)  
       aNumLastL = 1e9;

    mNameFile = aNameFile;
    if (! ExistFile(mNameFile))
    {
       MMVII_UserError(eTyUEr::eOpenFile,std::string("For file ") + mNameFile);
    }
    std::ifstream infile(mNameFile);

    std::string line;
    // mNbLineRead = 0;
    int aNumL = 0;
    while (std::getline(infile, line))
    {
         if ((aNumL>=aL0) && (aNumL< aNumLastL))
	 {
             const char * aC = line.c_str();
             bool  GoOn = true;
	     size_t aNbToken = 0;
	     while (GoOn)
	     {
                 while ( (isspace(*aC))  || (*aC==',')   ) aC++;
                 if ((*aC==aComment) || (*aC==0))
		 {
		     GoOn= false;
		     if ((aNbToken!=0) &&  (aNbToken != mTypes.size()))
		     {
                         MMVII_UserError(eTyUEr::eUnClassedError,"Bad nb token at line " + ToStr(aNumL) + " of file " + mNameFile);
		     }
		 }
		 else 
                 {
                      const char * aC0 = aC;
		      while (*aC && (!isspace(*aC)) && (*aC!=aComment)  && (*aC!=','))
                            aC++;

	              std::string aToken(aC0,size_t(aC-aC0));
		      std::string aNameField = mNameFields.at(aNbToken);
		      switch (mTypes.at(aNbToken))
		      {
		  	      case  eTypeField::eInt  : 
                                    AddVal(aNameField,aToken,mMapInt);
                              break;

		  	      case  eTypeField::eFloat  : 
                                    AddVal(aNameField,aToken,mMapFloat);
                              break;

		  	      case  eTypeField::eString  : 
                                    AddVal(aNameField,aToken,mMapString);
                              break;


			      default :
                              break;
		      }


	              aNbToken ++;

		 }
	     }
	 }
	 if (mDebug)
	    StdOut() << "-------------------------------------------\n";
         aNumL++;
    }
}



   /* ********************************************************** */
   /*                                                            */
   /*                 cAppli_ImportLines                         */
   /*                                                            */
   /* ********************************************************** */

class cAppli_ImportLines : public cMMVII_Appli
{
     public :
        cAppli_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;

        std::vector<std::string>  Samples() const override;
     private :

	cPhotogrammetricProject  mPhProj;

	// Mandatory Arg
	std::string              mNameFile;
	std::string              mFormat;

	// Optionall Arg
	int                        mL0;
	int                        mLLast;
	char                        mComment;

	//   Format specif
	std::string              mNameIm;
	std::string              mNameX1;
	std::string              mNameY1;
	std::string              mNameX2;
	std::string              mNameY2;
	std::string              mNameSigma;
	std::string              mSpecFormat;

};

cAppli_ImportLines::cAppli_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec) :
   cMMVII_Appli    (aVArgs,aSpec),
   mPhProj         (*this),
   mL0             (0),
   mLLast          (-1),
   mComment        (0),
   mNameIm         ("Im"),
   mNameX1         ("X1"),
   mNameY1         ("Y1"),
   mNameX2         ("X2"),
   mNameY2         ("Y2"),
   mNameSigma      ("Sigma"),
   mSpecFormat     (mNameIm+mNameX1+mNameY1+mNameX2+mNameY2+mNameSigma)
{
	// std::map<std::string,int>  aMap{{"2",2}};
}

cCollecSpecArg2007 & cAppli_ImportLines::ArgObl(cCollecSpecArg2007 & anArgObl) 
{
    return anArgObl
	      <<  Arg2007(mNameFile ,"Name of Input File")
	      <<  Arg2007(mFormat   ,"Format of file as for ex \"" + mSpecFormat + "\"")
              // <<  mPhProj.DPPointsMeasures().ArgDirOutMand()
           ;
}

cCollecSpecArg2007 & cAppli_ImportLines::ArgOpt(cCollecSpecArg2007 & anArgOpt) 
{
    return anArgOpt
       << AOpt2007(mL0,"NumL0","Num of first line to read",{eTA2007::HDV})
       << AOpt2007(mLLast,"NumLast","Num of last line to read (-1 if at end of file)",{eTA2007::HDV})
       << AOpt2007(mComment,"Comment","Carac for comment")
       << mPhProj.ArgSysCo()
    ;
}


int cAppli_ImportLines::Exe()
{
	// end init & check
    mPhProj.FinishInit();

    cNewReadFilesStruct aNRFS(mFormat,mSpecFormat);

    aNRFS.ReadFile(mNameFile,mL0,mLLast ,mComment);

#if (0)
void cNewReadFilesStruct::
    MMVII_INTERNAL_ASSERT_tiny(CptSameOccur(mFormat,"XYZij")==1,"Bad format vs NIXY");

       //  read file
    cReadFilesStruct aRFS(mNameFile, mFormat, mL0, mLLast, mComment);
    aRFS.Read();
       // create structur to import in MMVII representation
    if (mAddIm2NamePt)
        mNameGCP = mNameGCP + mNameImage;
    cSetMesGCP aSetGCP(mNameGCP);
    cSetMesPtOf1Im aSetIm(mNameImage);

       // parse all object to push them in low MVVII
    for (int aKObj=0 ; aKObj<aRFS.NbRead() ; aKObj++)
    {
         const cPt2dr & aP2 = aRFS.Vij().at(aKObj);
         const cPt3dr & aP3 = aRFS.VXYZ().at(aKObj);

	 std::string aNamePt = std::string("Pt_") + ToStr(aKObj);
         if (mAddIm2NamePt)
         {
            aNamePt = aNamePt + "_" + mNameImage;
         }
	 cMes1GCP aMesGCP(aP3,aNamePt,1.0);
	 cMesIm1Pt aMesIm(aP2,aNamePt,1.0);

	 aSetGCP.AddMeasure(aMesGCP);
         aSetIm.AddMeasure(aMesIm);
    }

       // save object
    mPhProj.SaveGCP(aSetGCP);
    mPhProj.SaveMeasureIm(aSetIm);

    if (mPhProj.SysCoIsInit())
        mPhProj.SaveStdCurSysCo(false);
#endif

    return EXIT_SUCCESS;
}

std::vector<std::string>  cAppli_ImportLines::Samples() const
{
   return 
   {
          "MMVII ImportM32 verif_1B.txt SjiXYZ XingB NumL0=13 NumLast=30 NameIm=SPOT_1B.tif"
   };
}


tMMVII_UnikPApli Alloc_ImportLines(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppli_ImportLines(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_ImportLines
(
     "ImportLine",
      Alloc_ImportLines,
      "Import/Convert Set of lines extracted",
      {eApF::Lines},
      {eApDT::Lines},
      {eApDT::Lines},
      __FILE__
);
#if (0)
#endif

}; // MMVII


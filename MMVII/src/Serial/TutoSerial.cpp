#include "MMVII_2Include_Serial_Tpl.h"
/*
#include "MMVII_Bench.h"
#include "MMVII_Class4Bench.h"

#include "MMVII_Geom2D.h"
#include "MMVII_PCSens.h"
#include "Serial.h"
#include "MMVII_MeasuresIm.h"
*/


/** \file TutoSerial.cpp
    \brief Tuto file for demonstration of some serialization mecanism.

    The primary "pitch" of this tutorial is the folowing :

        - we have a class people with a certain number of field
	- people can be created in a process and used in another process 
	- so we need to read an write people on files,  the global method "SaveInFile" and "ReadFromFile"
	  can be used for that provided that  people can describe itself using the method "AddData"

    Now in real life, with conception of "big" object, it will happen that object can evolved with more complexity
    and that fields no initially present must be added.  The key is that dont want to loose the object that
    have been created before.  The c++ "std::optionnal" offers such option inside the code, and we checked
    that for tagged format, it works (i.e. we can read "old" files).

    Another important requirement is that if, in the same process, we read, at different part of the code,
    the same file we get "physically" the same object


              -----------------------------------------------------

    After seeing the basic mecanism of serialization for reading/writing data,  two
    more evolved question are adressed :
    
        - use of optionnal in serialization to solve backward compatibility
	- use of method for reading "remanent" object

    We also study some mecanism of memory checking :

         -  cMemCheck : for checking that all object allocated are destroyed
	 -  cObj2DelAtEnd for deleting object at the end of the process



*/

namespace MMVII
{

static constexpr int Id1 = 90867;
static constexpr int Id2 = 19807;

class cPeople :  
          	public cMemCheck,
          	public cObj2DelAtEnd
{
   public :

       cPeople(); 
       ~cPeople(); 

       static cPeople * New(int anId,const std::string & aName,double aSalary);
       static cPeople * New(int anId,const std::string & aName,double aSalary,const cPt2dr &aPos);
       void AddData(const cAuxAr2007 &);

       void SaveAuto() const;
       static cPeople * Read1(int anId);
       static cPeople * Read2(int anId);
       void Show() const;
       int *Buf() ;
    private :
       cPeople(int anId,const std::string & aName,double aSalary);
       cPeople(int anId,const std::string & aName,double aSalary,const cPt2dr &aPos);

       static std::string NameFile(int anId) {return "Id_"+ToStr(anId) + ".xml";}

       int                   mIdNumber;
       std::string           mName;
       double                mSalary;
       std::optional<cPt2dr> mPos;
       int                   mBuf[4]; // to test overwrite
};

cPeople::cPeople() 
{
}

cPeople::~cPeople() 
{
}

cPeople::cPeople(int anId,const std::string & aName,double aSalary) :
   mIdNumber (anId),
   mName     (aName),
   mSalary   (aSalary)
{
}

cPeople::cPeople(int anId,const std::string & aName,double aSalary,const cPt2dr &aPos) :
    cPeople( anId,aName,aSalary)
{
	mPos  = aPos;
}

void cPeople::Show() const
{
     StdOut() << " Id=" << mIdNumber << " Name="<< mName << " Salary=" << mSalary ;
     if (mPos.has_value())   StdOut() << " Pos=" << mPos.value() ;
     StdOut() << std::endl;
}

int * cPeople::Buf()  {return mBuf;}


cPeople * cPeople::New(int anId,const std::string & aName,double aSalary) {return new cPeople(anId,aName,aSalary);}
cPeople * cPeople::New(int anId,const std::string & aName,double aSalary,const cPt2dr &aPos) {return  new cPeople(anId,aName,aSalary,aPos);}


void cPeople::AddData(const cAuxAr2007 & anAux)
{
    MMVII::AddData(cAuxAr2007("Id",anAux)   , mIdNumber);
    MMVII::AddData(cAuxAr2007("Name",anAux) , mName);
    MMVII::AddData(cAuxAr2007("Salary",anAux) , mSalary);
    MMVII::AddOptData(anAux,"Pos",mPos);
}
void AddData(const cAuxAr2007 & anAux,cPeople & aPeople) {aPeople.AddData(anAux);}


void cPeople::SaveAuto() const
{
     SaveInFile(*this, NameFile(mIdNumber) );
}

cPeople * cPeople::Read1(int anId)
{
    cPeople * aRes = new cPeople;
    ReadFromFile(*aRes,NameFile(anId));

    return aRes;
}

cPeople * cPeople::Read2(int anId)
{
   return SimpleRemanentNewObjectFromFile<cPeople>(NameFile(anId));
}


/* *********************************************************** */
/*                                                             */
/*                   cAppliTutoSerial                          */
/*                                                             */
/* *********************************************************** */

class cAppliTutoSerial : public cMMVII_Appli
{
     public :

        cAppliTutoSerial(const std::vector<std::string> & aVArgs,const cSpecMMVII_Appli & aSpec);

     private :
        int Exe() override;
        cCollecSpecArg2007 & ArgObl(cCollecSpecArg2007 & anArgObl) override ;
        cCollecSpecArg2007 & ArgOpt(cCollecSpecArg2007 & anArgOpt) override ;
};


cAppliTutoSerial::cAppliTutoSerial
(
    const std::vector<std::string> & aVArgs,
    const cSpecMMVII_Appli & aSpec
) :
   cMMVII_Appli   (aVArgs,aSpec)
{
}

cCollecSpecArg2007 & cAppliTutoSerial::ArgObl(cCollecSpecArg2007 & anArgObl)
{
    return anArgObl
   ;
}


cCollecSpecArg2007 & cAppliTutoSerial::ArgOpt(cCollecSpecArg2007 & anArgOpt)
{
   return
                  anArgOpt
          ;
}



int  cAppliTutoSerial::Exe()
{
    
   //  [1]  Create object and see what is on disk
   {
       cPeople * aP1 = cPeople::New(Id1,"Dupont",23.5);  // Create an object
       aP1->SaveAuto();  // Save it to file

       for (int aK=0 ; aK<4; aK++)  aP1->Buf()[aK] = aK; // OK
       // for (int aK=0 ; aK<=4; aK++) aP1->Buf()[aK] = aK; // NOT OK

       delete aP1;  //  if no  delete , memory leak will be detected
   }
   //  [2]  Test that we are able to re-read it
   {
         StdOut() << " ------------- READ INIT ------------------" << std::endl;
         std::unique_ptr<cPeople > aP1 (cPeople::Read1(Id1));
	 aP1->Show();
   }

   // [3]  Now we add a position, we  write it
   {
        std::shared_ptr<cPeople > aP2 (cPeople::New(Id2,"Durand",33.5,cPt2dr(2,3)));
        aP2->SaveAuto();
   }

   // [4]  Test that we can read the "old" object and the "new" one
   {
         StdOut() << " ------------- READ AGAIN ------------------" << std::endl;
         std::unique_ptr<cPeople > aP1 (cPeople::Read1(Id1));
         std::unique_ptr<cPeople > aP2 (cPeople::Read1(Id2));
	 aP1->Show();
	 aP2->Show();
   }

   // [5]  Check that for now, two object with same ident are diffent
   {
          cPeople * aP1A = cPeople::Read1(90867);
          cPeople * aP1B = cPeople::Read1(90867);

	  StdOut()  <<  "Is P1A same object than P1B?  " << ((aP1A==aP1B) ? " YES " : " NO") << std::endl;
	  delete aP1A;
	  delete aP1B;
   }


   // [6]  Now with Read2  that use "SimpleRemanentObjectFromFile" they are the same
   {
          cPeople * aP1A = cPeople::Read2(90867);
          cPeople * aP1B = cPeople::Read2(90867);

	  StdOut()  <<  "Is P1A same object than P1B?  " << ((aP1A==aP1B) ? " YES " : " NO") << std::endl;
	 
	  //  delete aP1A;  we must not delete them, as many part can use it, it's the application responsability to delete them at end
	  // delete aP1B;
   }

   return EXIT_SUCCESS;
}

/* *********************************************************** */
/*                                                             */
/*                           ::                                */
/*                                                             */
/* *********************************************************** */


tMMVII_UnikPApli Alloc_cAppliTutoSerial(const std::vector<std::string> &  aVArgs,const cSpecMMVII_Appli & aSpec)
{
   return tMMVII_UnikPApli(new cAppliTutoSerial(aVArgs,aSpec));
}

cSpecMMVII_Appli  TheSpec_TutoSerial
(
     "TutoSerial",
      Alloc_cAppliTutoSerial,
      "Tutorial for serialization",
      {eApF::Project,eApF::Test},
      {eApDT::None},
      {eApDT::Xml},
      __FILE__
);

/*
*/




};

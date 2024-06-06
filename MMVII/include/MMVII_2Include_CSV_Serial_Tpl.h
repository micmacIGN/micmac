#ifndef  _MMVII_Serial_CSV_Tpl2Inc_H_
#define  _MMVII_Serial_CSV_Tpl2Inc_H_

/** \file MMVII_2Include_CSV_Serial_Tpl.h
    \brief Contains template definition  for CSV serialisation

*/

#include "MMVII_DeclareCste.h"
#include "MMVII_Stringifier.h"
#include "MMVII_Matrix.h"


namespace MMVII
{

class cOMakeTreeAr;
class cBaseCVSFile
{
    public :
    protected :
        cBaseCVSFile();
        ~cBaseCVSFile();

    private :
        cOMakeTreeAr *           mArTreeOut;
    protected :
        std::vector<std::string> mHeader;
        cAr2007*                 mArOut;
};
template <class Type>  class cCSVFile : public cBaseCVSFile
{
    public :
    protected :
       cCSVFile() :
         cBaseCVSFile ()
         {
            mArOut->SetSpecif(true);
            Type aT0{};
            AddData(cAuxAr2007("CSV",*mArOut,eTAAr::eStd),aT0);
            mArOut->PutArchiveIn(&mHeader);
            mArOut->SetSpecif(false);
         }
};

void PutLineCSV(cMMVII_Ofs &,const std::vector<std::string>  &) ;

template <class Type>  class cOCSVFile :  public cCSVFile<Type>
{
     public  :
          cOCSVFile<Type>(const std::string & aName,bool WithHeader) :
              cCSVFile<Type> (),
              mOfs           (aName,eFileModeOut::CreateText)
          {
              if (WithHeader)
                  PutLineCSV(mOfs,this->mHeader);
          }
          void  AddObj(const Type & anObj)
          {
               std::vector<std::string> aVS;
               AddData(cAuxAr2007("CSV",*(this->mArOut),eTAAr::eStd),const_cast<Type&>(anObj));
               this->mArOut->PutArchiveIn(&aVS);
               PutLineCSV(mOfs,aVS);
          }
     private  :
          cMMVII_Ofs mOfs;
};

template <class Type>  class cICSVFile :  public  cCSVFile<Type>,
                                          public  cAr2007
{
     public  :
          cICSVFile<Type>(const std::string & aName,bool WithHeader) :
              cCSVFile<Type> (),
              cAr2007(true,false,false) ,   // bool InPut,bool Tagged,bool Binary
              mIfs           (aName,eFileModeIn::Text),
              mNbCol         (this->mHeader.size())
          {
              if (WithHeader)
              {
                  ReadLine();
                  if (mBufLine!=  this->mHeader)
                  {
                      StdOut() << "Header got      -> " << mBufLine << std::endl;
                      StdOut() << "Header Expected -> "<< this->mHeader << std::endl;

                      MMVII_UnclasseUsEr("CSV FILE, header does not much what's excpected for type");
                  }
              }
          }

          void ReadFromFile(std::vector<Type> & aRes)
          {
              aRes.clear();
              while (ReadLine())
              {
                  Type anObj{};
                  AddData(cAuxAr2007("CSV",*this,eTAAr::eStd),anObj);
                  aRes.push_back(anObj);
              }
	      aRes.shrink_to_fit();
          }

     private  :

          template <class TypeVal>  void RawAddGen(TypeVal & aVal)
          {
                  aVal = cStrIO<TypeVal>::FromStr(mBufLine.at(mCurInd++));
          }

          void RawAddDataTerm(int &    anI  ) override {RawAddGen(anI);}
          void RawAddDataTerm(size_t & aSize) override {RawAddGen(aSize);}
          void RawAddDataTerm(double & aD   ) override {RawAddGen(aD);}
          void RawAddDataTerm(std::string &    aS)  override {RawAddGen(aS);}
          void RawAddDataTerm(cRawData4Serial  &    anI) override
          {
                  MMVII_INTERNAL_ERROR("No cRawData4Serial for CSV file");
          }

          bool ReadLine()
          {
              mCurInd=0;
              std::getline(mIfs.Ifs(), mLine);
              mBufLine = SplitString(mLine,",");

              if (mBufLine.empty())
                 return false;
              MMVII_INTERNAL_ASSERT_tiny(mBufLine.size()==mNbCol,"Bad number of column in  CSV file");
              return true;
          }

          std::string               mLine;
          std::vector<std::string>  mBufLine;
          cMMVII_Ifs                mIfs;
          size_t                    mNbCol; ///< number of theoreticall column
          size_t                    mCurInd; ///< number of theoreticall column
};



//======================================================================================

template <class Type>
    void ToCSV(const std::vector<Type> & aVObj,const std::string & aName,bool WithHeader)
{
       cOCSVFile<Type> aFile(aName,WithHeader);
       for (const auto & anObj : aVObj)
          aFile.AddObj(anObj);
}

template <class Type> void FromCSV(std::vector<Type>& aVect,const std::string & aNameFile,bool WithHeader)
{
	// StdOut() << "FromCSVFromCSV " << aNameFile << " " << ExistFile(aNameFile) << std::endl;
    cICSVFile<Type> aCvsIn1(aNameFile,WithHeader);
    aCvsIn1.ReadFromFile(aVect);
}




};

#endif //  _MMVII_Serial_CSV_Tpl2Inc_H_


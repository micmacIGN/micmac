#ifndef _MMVII_ReadFileStruct_
#define _MMVII_ReadFileStruct_
#include "MMVII_util_tpl.h"


/**
   \file cConvCalib.cpp  testgit

   \brief file for conversion between calibration (change format, change model) and tests
*/


namespace MMVII
{



/// Represent the type of the field that will be stored
enum class eRFS_TypeField
{
      eInt,
      eFloat,
      eString,
      eUnused,
      eBla       // For bla  bla
};


class  cNewReadFilesStruct
{
	 const std::map<std::string,std::vector<int> > &       StdMap(int*) const         {return  mMapInt;}
	 const std::map<std::string,std::vector<tREAL8> > &    StdMap(tREAL8*) const      {return  mMapFloat;}
	 const std::map<std::string,std::vector<std::string> >&StdMap(std::string*) const {return  mMapString;}


	 // value used to count occurence of "stared" atom like in "A1E*B" , this means that there can be any "E"
	 static const int StaredNumber = 10000;

     public :
         cNewReadFilesStruct(const std::string & aFormat,const std::string &  aSpecifFMand,const std::string &  aSpecifFTot);
	 
	 void ReadFile(const std::string & aNameFile,int aL0,int aLL,int aCom);

	 ///  Access to a vector of value from name of field (faster than GetValue, for very big file)
	 template <class Type> const std::vector<Type> &  GetVect(const std::string & aNameField) const
	 {
             const std::map<std::string,std::vector<Type> > & aMap = StdMap((Type*) nullptr);
	     // const auto & anIter = aMap.find(aNameField);
	     //typename std::map<std::string,std::vector<Type> >::const_iterator  anIter = aMap.find(aNameField);
	     const auto & anIter = aMap.find(aNameField);

             MMVII_INTERNAL_ASSERT_tiny(anIter!=aMap.end(),"GetVect failed for " +aNameField);

	     if (mDebug)
                StdOut() << "GetVect: " << aNameField << " " << anIter->second.size() << "\n";
	     const std::vector<Type>  & aV=  anIter->second;
	     return aV;
	 }

	 ///  Access to single value from name of field
         template <class Type> const Type &  GetValue(const std::string & aNameField,size_t aK) const
         {
                 MMVII_INTERNAL_ASSERT_tiny(MapBoolFind(mCptFields,aNameField),"No such field " +aNameField);
                 MMVII_INTERNAL_ASSERT_tiny(mCptFields.at(aNameField)==1,"Multiple field in line " +aNameField);

		 const Type& aRes = GetVect<Type>(aNameField).at(aK);
		 return aRes;
         }
	 
	 ///  Access to single value from name of field
         template <class Type> const Type &  GetKthValue(const std::string & aNameField,size_t aKLine,size_t aNumInLine) const
         {
                 MMVII_INTERNAL_ASSERT_tiny(MapBoolFind(mCptFields,aNameField),"No such field " +aNameField);
		 size_t aNbInL = mCptFields.at(aNameField);

                 MMVII_INTERNAL_ASSERT_tiny(aKLine<aNbInL,"No so many elem for "+ aNameField);

		 const Type& aRes = GetVect<Type>(aNameField).at(aKLine*aNbInL+aNumInLine);
		 return aRes;
         }




         const std::string & GetStr(const std::string & aNameField,size_t aK) const {return GetValue<std::string>(aNameField,aK);}
         const tREAL8 & GetFloat(const std::string & aNameField,size_t aK) const {return GetValue<tREAL8>(aNameField,aK);}
         const int & GetInt(const std::string & aNameField,size_t aK) const {return GetValue<int>(aNameField,aK);}

	 cPt2dr GetPt2dr(const std::string & aN1,const std::string & aN2,size_t aK) const
	 {
               return cPt2dr(GetFloat(aN1,aK),GetFloat(aN2,aK));
	 }

	 bool  FieldIsKnow(const std::string & aNameField) const
	 {
		 return BoolFind(mNameFields,aNameField);
	 }

	 size_t  ArrityField(const std::string & aNameField) const
	 {
		 return std::count(mNameFields.begin(),mNameFields.end(),aNameField);
	 }

	 size_t NbLineRead() const {return mNbLineRead;} /// Accessor
     private :
         
         // Add a value "Val" in field "Fied" to dictionnary of corresponding type ("String/Int/Float")
	 template <class Type> void AddVal
		                    (
				        const std::string & aNameField, 
				        const std::string & aNameValue, 
				        std::map<std::string,std::vector<Type> >& aMap
                                    )
	 {
		 if (mDebug) 
                     StdOut() <<  "   " << aNameField << " = [" << aNameValue  << "] T=("<<  cStrIO<Type>::msNameType << ")\n";
		 aMap[aNameField].push_back(cStrIO<Type>::FromStr(aNameValue));

	 }
	 

	 /// Check that the specif and the format are coherent
	 void Check(std::map<std::string,size_t> & aMap1,std::map<std::string,size_t> & aMap2,bool RefIsI1);

	 ///  return the type of a given name
	 eRFS_TypeField  TypeOfName(const std::string &);

	 /// Parse a format specif to split in token and count the occurence of each token
         static void  ParseFormat(bool IsSpec,const std::string & aFormat,std::map<std::string,size_t>  & aMap,std::vector<std::string> & );

         std::string                     mFormat;      ///< Format used to parse the file 
	 bool mDebug;

	 std::map<std::string,size_t>    mCptFields;   ///< count the occurence of each token in the format
	 std::vector<std::string>        mNameFields;  ///< list of token
	 std::vector<eRFS_TypeField>         mTypes;       ///< list of type of token
						       //
	 std::string                     mNameFile;


	 std::map<std::string,std::vector<int> >          mMapInt;
	 std::map<std::string,std::vector<tREAL8> >       mMapFloat;
	 std::map<std::string,std::vector<std::string> >  mMapString;

	 size_t                                           mNbLineRead;

};


}; //  MMVII

#endif //  _MMVII_ReadFileStruct_

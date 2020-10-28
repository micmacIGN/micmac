//extracted from private/files.h
#ifndef FILES_EXTRACT_H
#define FILES_EXTRACT_H

template <class Type>
void MakeFileXML(const Type & anObj,const std::string & aName,const std::string & aTagEnglob="")
{
   if (IsFileDmp(aName))
   {
       BinDumpObj(anObj,aName);
       return;
   }

   cElXMLTree * aTree = ToXMLTree(anObj);
   // FILE * aFp = Fopen(aName.c_str(),"w");
   // Fclose(aFp);
   if (aTagEnglob!="")
   {
      aTree = cElXMLTree::MakePereOf(aTagEnglob,aTree);
   }
   aTree->StdShow(aName);
   delete aTree;
}

#endif //SUPERPOSIMAGE_EXTRACT_H

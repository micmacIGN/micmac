#ifndef SYMBDER_NAMEALLOC_H
#define SYMBDER_NAMEALLOC_H

#include <string>
#include <vector>
#include <fstream>

#include "SymbolicDerivatives.h"

namespace  NS_SymbolicDerivative
{

class cGenNameAlloc
{
public:
    enum Command {RESET, ADD, GEN_FILE};

    // This is more or less a namespace ...
    cGenNameAlloc() = delete;

    // Desc of this methods is bellow
    static void Reset(void)
    {
        DoCommand(RESET);
    }

    static void Add(const std::string& aClassName, const std::string& aFileName)
    {
        DoCommand(ADD,aClassName,aFileName);
    }

    static void GenerateFile(const std::string& aFileName, const std::string& aHeaderIncludeSymbDer, const std::string& aIncludeDir)
    {
        DoCommand(GEN_FILE,aFileName,aHeaderIncludeSymbDer, aIncludeDir);
    }


private:
    // The trick is to have a static object in a .h
    // We put it in a function that does all the work on "demand"
    //   command RESET    => Clear the vector of generated Functions/File names. No Args.
    //   command ADD      => Push a new pair of Function/File name. Arg1: function name, Arg2: File Name
    //   command GEN_FILE => Generate a file defining cName2Calc::InitMapAlloc() to register all generated Functions
    //                 (see SymbDer_Common.h)
    //                 Arg1: Full name of file to generate,
    //                 Arg2: Include path for SymbDer_Common.h,
    //                 Arg3: Common Directory path for all generated functions files
    //                   (can be empty string, if the generated file is in the same dir than the generated function files).
    static void DoCommand(Command aCommand, const std::string& aArg1="", const std::string& aArg2="",std::string aArg3="")
    {
        static std::vector<std::pair<std::string,std::string>> generatedFuncs;

        switch (aCommand) {
        case RESET:
            generatedFuncs.clear();
            return;
        case ADD:
            for (const auto &[name,file] : generatedFuncs) {
                if (name == aArg1)
                    UserSError("In cGenNameAlloc::Add() : Function name '" + name + "' already registered","");
                if (file == aArg2)
                    UserSError("In cGenNameAlloc::Add() : File name '" + file + "' already used for '" + name + "'","");
            }
            generatedFuncs.emplace_back(std::make_pair(aArg1,aArg2));
            return;
        case GEN_FILE:
            std::ofstream aOs(aArg1);
            if (!aOs)
                UserSError("In cGenNameAlloc::GenerateFile() : Can't create file '" + aArg1 +"'","");
            if (aArg3 != "" && aArg3[aArg3.size() -1] != '/')
                aArg3 += "/";
            aOs << "#include \"" << aArg2 << "\"\n";
            for (const auto& f: generatedFuncs)
                aOs << "#include \"" << aArg3 << f.second << ".h\"\n";
            aOs << "\n";
            aOs << "namespace NS_SymbolicDerivative {\n";
            aOs << "\n";
            aOs << "void cName2CalcRegisterAll(void)\n";
            aOs << "{\n";
            aOs << "  static bool firstCall=true;\n";
            aOs << "\n";
            aOs << "  if (! firstCall)\n";
            aOs << "    return;\n";
            aOs << "  firstCall = false;\n";
            for (const auto& f: generatedFuncs)
                aOs << "  " << f.first << "::Register();\n";
            aOs << "}\n";
            aOs << "}\n  // namespace NS_SymbolicDerivative";
            if (!aOs)
                UserSError("In cGenNameAlloc::GenerateFile() : Error in writing file '" + aArg1 +"'","");
            return;
        }
    }

};

} // namespace  NS_SymbolicDerivative



#endif // SYMBDER_NAMEALLOC_H

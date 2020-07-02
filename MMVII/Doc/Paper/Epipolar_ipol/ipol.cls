%-------------------------------------------------------------------------------
% ipol.cls
% LaTeX class for IPOL articles
%
% version 0.5.2 - 2015.11.7 - jose luis lisani 
%    > change in format of doi links (from http://dx.doi.org to https://doi.org)
%       following the indications of the Crossref organization
% version 0.5.1 - 2015.6.16 - rafael grompone von gioi
%    > small corrections to page layout to avoid "Overfull \vbox" messages
%      due to a zero height of the header
% version 0.5 - 2014.7.1 - rafael grompone von gioi
%    > abstract included as PDF metadata
%    > vector version of IPOL logo
% version 0.4.3 - 2014.4.29 - rafael grompone von gioi
%    > last page reference bug corrected (in some cases, the first page footer
%      with the article reference would not print the correct last page number.
% version 0.4.2 - 2014.3.18 - nicolas limare
%    > use SIAM style for the standard ipol citation
%    > insert custom PDF metatata: IPOL.Volume and IPOL.DOI
% version 0.4.1 - 2014.3.8 - rafael grompone von gioi
%    > new command to specify the volume number
% version 0.4 - 2013.7.19 - rafael grompone von gioi
%    > the IPOL LaTeX class version label was moved a little so it is
%      not cut by the margins of the print area of printers
%    > copyediting commands were removed
% version 0.3.2 - 2013.7.4 - rafael grompone von gioi
%    > verification added that the right version of package 'hyperref' is
%      available. otherwise, the PDF metadata is not generated
%    > the IPOL LaTeX class version is added to the PDF 'Creator' metadata
%    > when the document is a preprint, its DOI and link is set to
%      https://doi.org/10.5201/ipol so it points to the IPOL journal
%    > modification so IPOL LaTeX class version is defined only once
%    > some minor source code arrangements
% version 0.3.1 - 2013.6.21 - rafael grompone von gioi
%    > bug corrected: pages in article reference at footer should not be links
%    > bug corrected: article's reference in footer absent if first page even
% version 0.3 - 2013.5.1 - rafael grompone von gioi
%    > "\bf" replaced with the more standard one "\bfseries"
%    > typo corrected
%    > bygone 'pagecolor' option of package 'hyperref' removed
%    > copyediting commands added
%    > IPOL logo linked to the website
%    > the PREPRINT label, when needed, is placed below title and authors
%    > \relax added after \ProcessOptions, see The LaTeX Companion 2ed. p.882
%    > release date and version added to the class definition
%    > \vbox command replaced by the more standard one \parbox
%    > keywords handling added
%    > PDF metadata generation
%    > article's date replaced by submission, accepted and publication dates
%    > submission and accepted dates are printed in the header
%    > new command \ipolAuthorMark to set the affiliations markers in a way
%      that can be removed when setting the PDF metadata
%    > abstract environment renamed to ipolAbstract. This make the naming
%      more coherent and allows to use the package "environ" (for some reason
%      it is not able to redefine the abstract environment).
%    > in ipolAbstract, ipolCode and ipolSupp, the 'quote' environment
%      was replace by the 'quotation' environment. This indents the
%      paragraphs except for the initial one.
%    > SIIMS companion article command added
%    > labels added to the first and last page numbers
%    > headers added
%    > first page footer contains the article citation
%    > a command added to force the citation text
%    > IPOL LaTeX class version is printed in the upper left side of first page
% version 0.2 - 2012.6.4 - rafael grompone von gioi
%    > the logo image is now called "ipol_logo".
%    > affiliations command added
%    > "DRAFT" label became "PREPRINT"
%    > margins copied from "fullpage" without using it
%    > "\ipolDraftLink" command became "\ipolPreprintLink"
%    > modifications to the paper's header
%    > "\href" command was modified in order that links will include a footnote
%      with the address, to get the links in printed form too.
%    > links' color modified.
%    > compilation date added to the Preprint label
%    > modifications to the mechainsm that generates the title of the article.
% version 0.1 - 2012.1.27 - rafael grompone von gioi
%-------------------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\newcommand{\ipolClassVersion}{2015/06/16 v0.5.1 IPOL article class}
\ProvidesClass{ipol}[\ipolClassVersion]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[a4paper,12pt,twoside]{article}

% Page Layout
\setlength{\textwidth}{\paperwidth}
\addtolength{\textwidth}{-3cm} % margins
\setlength{\oddsidemargin}{1.5cm}
\addtolength{\oddsidemargin}{-1in} % remove default 1 inch offset
\setlength{\evensidemargin}{\oddsidemargin}
\setlength{\topmargin}{0.87cm}
\addtolength{\topmargin}{-1in} % remove default 1 inch offset
\setlength{\headheight}{0.4cm}
\setlength{\headsep}{0.925cm}
\setlength{\textheight}{\paperheight}
\addtolength{\textheight}{-\headheight}
\addtolength{\textheight}{-\headsep}
\addtolength{\textheight}{-\footskip}
\addtolength{\textheight}{-2.37cm} % margins

% Headers
% Definition of standard ipol citation
\newcommand{\ipolCitation}{%
  \textsc{\ipolAuthors}, \emph{\ipolTitle},
  Image Processing On Line, \ipolVolume\ (\ipolPublicationYear),
  pp.~\pageref*{ipol:class:first:page}--\pageref*{LastPage}.
  \ipolLink}
% Command to force citation
\newcommand{\ipolForceCitation}[1]{\renewcommand{\ipolCitation}{#1}}
% Definition of ipol page style
\newcommand{\ps@ipol}{%
  \renewcommand{\@evenhead}{\hfil\scriptsize\textsc{\ipolAuthors}\hfil}%
  \renewcommand{\@oddhead}{\hfil\scriptsize\textsc{\ipolTitle}\hfil}%
  \renewcommand{\@evenfoot}{\hfil\textrm{\thepage}\hfil}%
  \renewcommand{\@oddfoot}{\@evenfoot}}
% Definition of ipolFirstPage page style
\newcommand{\ps@ipolFirstPage}{%
  \renewcommand{\@oddhead}{}%
  \renewcommand{\@evenhead}{}%
  \renewcommand{\@oddfoot}{%
    \@ifundefined{ipolIsPreprint}{%
      \begin{minipage}[t]{\linewidth}
        \sf\scriptsize\ipolCitation
      \end{minipage}}{}}
  \renewcommand{\@evenfoot}{%
    \@ifundefined{ipolIsPreprint}{%
      \begin{minipage}[t]{\linewidth}
        \sf\scriptsize\ipolCitation
      \end{minipage}}{}}}
\pagestyle{ipol}

% Links package and colors definition
\RequirePackage{color}
\definecolor{link}{rgb}{0.240,0.194,0.431}
\RequirePackage[colorlinks,
                linkcolor=link,
                anchorcolor=link,
                citecolor=link,
                filecolor=link,
                menucolor=link,
                urlcolor=link]{hyperref}
% if hyperref version is old, no PDF metadata will be generated:
\@ifpackagelater{hyperref}{2010/01/11}{}{%
    \newcommand{\ipolNoPDFMetadata}{}%
    \ClassWarningNoLine{ipol}{hyperref version previous to 2010/01/11,
                              PDF metadata will not be generated}}
% The following definitions modify the "\href" command to
% also include a footnote with the address, so it is also
% accessible in printed form. The command "\basehref" is
% created in cases where the footnote wants to be avoided.
\newcommand*\basehref[2]{}
\let\basehref\href
\renewcommand*\href[2]{\basehref{#1}{#2}\footnote{\url{#1}}}

% IPOL environments
% Abstract
% if the 'environ' package is available, use it for pdf abstract metadata
\newcommand{\ipolAbstractText}{}
\IfFileExists{environ.sty}{%
  \RequirePackage{environ}
  \NewEnviron{ipolAbstract}{\small%
                            \begin{center}%
                              {\bfseries Abstract\vspace{-.5em}}%
                            \end{center}%
                            \begin{quotation}%
                              \noindent\ignorespaces\BODY%
                            \end{quotation}%
                            \@ifundefined{ipolIsPreprint}{
                              \@ifundefined{ipolNoPDFMetadata}{
                                \ClassWarningNoLine{ipol}{If an error follows,
                                  the abstract may have many paragraphs. To
                                  work correctly, IPOL class requires the
                                  abstract to be only ONE paragraph}%
                                \hypersetup{pdfinfo={Subject={\BODY}}}%
                              }{}%
                            }{}%
                           }{}%
}{%
  \newenvironment{ipolAbstract}{\small%
                                \begin{center}%
                                {\bfseries Abstract\vspace{-.5em}}%
                                \end{center}%
                                \begin{quotation}%
                                \noindent\ignorespaces}{\end{quotation}}
}
% Code section
\newenvironment{ipolCode}{\small%
                          \begin{center}%
                          {\bfseries Source Code\vspace{-.5em}}%
                          \end{center}%
                          \begin{quotation}%
                          \noindent\ignorespaces}{\end{quotation}}
% Supplementary material section
\newenvironment{ipolSupp}{\small%
                          \begin{center}%
                          {\bfseries Supplementary Material\vspace{-.5em}}%
                          \end{center}%
                          \begin{quotation}%
                          \noindent\ignorespaces}{\end{quotation}}

% IPOL command to set Keywords
\newcommand{\ipolKeywords}[1]{%
  \begin{quote}%
  \noindent\ignorespaces\small\textbf{Keywords:} #1%
  \end{quote}%
  \@ifundefined{ipolNoPDFMetadata}{\hypersetup{pdfinfo={Keywords={#1}}}}{}
}

% command to declare SIIMS companion article
\newenvironment{ipolSIIMS}{%
  \begin{center}%
  \begin{minipage}{0.67\linewidth}%
    \raisebox{-\height}{%
      \basehref{http://www.siam.org/journals/siims.php}%
               {\includegraphics[width=0.22\linewidth]{siims_logo}}}%
    \hspace{0.01\linewidth}%
    \begin{minipage}[t]{0.77\linewidth}\vspace{0pt}%
      \scriptsize\sf\noindent\ignorespaces}%
  {\end{minipage}\end{minipage}\end{center}\bigskip}

% IPOL data
\newcommand{\ipolTitle}{title}
\newcommand{\ipolAuthors}{authors}
\newcommand{\ipolAffiliations}{}
\newcommand{\ipolSubmissionYear}{YYYY}
\newcommand{\ipolSubmissionMonth}{MM}
\newcommand{\ipolSubmissionDay}{DD}
\newcommand{\ipolAcceptedYear}{YYYY}
\newcommand{\ipolAcceptedMonth}{MM}
\newcommand{\ipolAcceptedDay}{DD}
\newcommand{\ipolPublicationYear}{YYYY}
\newcommand{\ipolPublicationMonth}{MM}
\newcommand{\ipolPublicationDay}{DD}
\newcommand{\ipolID}{XXXXXXXX}
\newcommand{\ipolVolume}{n}
\newcommand{\ipolDOI}{\@ifundefined{ipolIsPreprint}%
                        {10.5201/ipol.\ipolPublicationYear.\ipolID}%
                        {10.5201/ipol}}
\newcommand{\ipolLink}{https://doi.org/\ipolDOI}

% IPOL data set commands
\newcommand{\ipolSetTitle}[1]{\renewcommand{\ipolTitle}{#1}%
                              \newcommand{\ipolTitleDefined}{}}
\newcommand{\ipolSetAuthors}[1]{\renewcommand{\ipolAuthors}{#1}%
                                \newcommand{\ipolAuthorsDefined}{}}
\newcommand{\ipolSetAffiliations}[1]{\renewcommand{\ipolAffiliations}{#1}%
                                     \newcommand{\ipolAffiliationsDefined}{}}
\newcommand{\ipolSetID}[1]{\renewcommand{\ipolID}{#1}%
                           \newcommand{\ipolVolumeDefined}{}}
\newcommand{\ipolSetVolume}[1]{\renewcommand{\ipolVolume}{#1}%
                               \newcommand{\ipolIDDefined}{}}
\newcommand{\ipolPreprintLink}[1]{\renewcommand{\ipolLink}{#1}%
                                  \newcommand{\ipolPreprint}{}}
\newcommand{\ipolAuthorMark}[1]{}

% set submission date commands
\newcommand{\ipolSetSubmissionYear}[1]{\renewcommand{\ipolSubmissionYear}{#1}%
                            \newcommand{\ipolSubmissionYearDefined}{}}
\newcommand{\ipolSetSubmissionMonth}[1]{\renewcommand{\ipolSubmissionMonth}{#1}%
                            \newcommand{\ipolSubmissionMonthDefined}{}}
\newcommand{\ipolSetSubmissionDay}[1]{\renewcommand{\ipolSubmissionDay}{#1}%
                            \newcommand{\ipolSubmissionDayDefined}{}}

% set accepted date commands
\newcommand{\ipolSetAcceptedYear}[1]{\renewcommand{\ipolAcceptedYear}{#1}%
                            \newcommand{\ipolAcceptedYearDefined}{}}
\newcommand{\ipolSetAcceptedMonth}[1]{\renewcommand{\ipolAcceptedMonth}{#1}%
                            \newcommand{\ipolAcceptedMonthDefined}{}}
\newcommand{\ipolSetAcceptedDay}[1]{\renewcommand{\ipolAcceptedDay}{#1}%
                            \newcommand{\ipolAcceptedDayDefined}{}}

% set publication date commands
\newcommand{\ipolSetPublicationYear}[1]{\renewcommand{\ipolPublicationYear}{#1}%
                            \newcommand{\ipolPublicationYearDefined}{}}
\newcommand{\ipolSetPublicationMonth}[1]{\renewcommand{\ipolPublicationMonth}%
                            {#1}\newcommand{\ipolPublicationMonthDefined}{}}
\newcommand{\ipolSetPublicationDay}[1]{\renewcommand{\ipolPublicationDay}{#1}%
                            \newcommand{\ipolPublicationDayDefined}{}}

% Print IPOL Header and Title
\RequirePackage{graphicx}
\AtBeginDocument{%
% Decide between Preprint or Publication mode. This must be done at
% the beginning of the document, when all IPOL definitions were already done.
\@ifundefined{ipolTitleDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolAuthorsDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolSubmissionYearDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolSubmissionMonthDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolSubmissionDayDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolAcceptedYearDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolAcceptedMonthDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolAcceptedDayDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolPublicationYearDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolPublicationMonthDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolPublicationDayDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolIDDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolVolumeDefined}{\newcommand{\ipolIsPreprint}{}}{%
\@ifundefined{ipolPreprint}{}{\newcommand{\ipolIsPreprint}{}}}}}}}}}}}}}}}
%
% print IPOL header
\basehref{https://doi.org/10.5201/ipol}%
  {\includegraphics[width=.32\linewidth]{ipol_logo}}%
\hspace{1em}%
\parbox[b]{0.6\linewidth}{\scriptsize\sf\noindent%
Published in \basehref{https://doi.org/10.5201/ipol}{Image Processing On Line}
on %
\@ifundefined{ipolIsPreprint}%
  {\ipolPublicationYear--%
     \two@digits{\ipolPublicationMonth}--%
     \two@digits{\ipolPublicationDay}}%
  {YYYY--MM--DD}.\\
Submitted on %
\@ifundefined{ipolIsPreprint}%
  {\ipolSubmissionYear--%
     \two@digits{\ipolSubmissionMonth}--%
     \two@digits{\ipolSubmissionDay}, }%
  {YYYY--MM--DD, }%
accepted on
\@ifundefined{ipolIsPreprint}%
  {\ipolAcceptedYear--%
     \two@digits{\ipolAcceptedMonth}--%
     \two@digits{\ipolAcceptedDay}}%
  {YYYY--MM--DD}.\\
ISSN 2105--1232 \copyright\ \ipolPublicationYear\ IPOL \& the authors
\basehref{http://creativecommons.org/licenses/by-nc-sa/3.0/}{CC--BY--NC--SA}\\
This article is available online with supplementary materials,\\
software, datasets and online demo at\\
\url{\ipolLink}}

% Print IPOL class version
\RequirePackage{color}
\RequirePackage{rotating}
\definecolor{ipolVersion}{gray}{0.8}
\hspace{-16mm}
\begin{rotate}{90}
{\color{ipolVersion}\tiny\ipolClassVersion}
\end{rotate}
\medskip
\bigskip
% print title & authors
\begin{center}
\Large\textbf{\ipolTitle}\\
\bigskip
\renewcommand{\ipolAuthorMark}[1]{$^{#1}$}
\large\ipolAuthors\\
\bigskip
\@ifundefined{ipolAffiliationsDefined}{}{%
  \footnotesize\ipolAffiliations\\%
  \bigskip%
}
\@ifundefined{ipolIsPreprint}{}{%
  \bigskip%
  {\sf\Large\color{red}PREPRINT \@date}%
}
\bigskip
\end{center}
% set PDF metadata
\@ifundefined{ipolIsPreprint}%
  {\newcommand{\ipolMetadataDate}{D:\ipolPublicationYear%
                                  \two@digits{\ipolPublicationMonth}%
                                  \two@digits{\ipolPublicationDay}000000}}%
  {\newcommand{\ipolMetadataDate}{}}%
\renewcommand{\ipolAuthorMark}[1]{}
\@ifundefined{ipolNoPDFMetadata}{%
  \hypersetup{
    pdfinfo={
      Title={\ipolTitle},
      Author={\ipolAuthors},
      CreationDate={\ipolMetadataDate},
      ModDate={ },
      Creator={\ipolClassVersion},
      IPOL.Volume={\ipolVolume},
      IPOL.DOI={\ipolLink},
    }
  }
}{}
\label{ipol:class:first:page} % save first page number
\thispagestyle{ipolFirstPage} % first page style
}% end of \AtBeginDocument

\RequirePackage{lastpage} % used to save a reference to the last page
                          % to prevent other packages including commands
                          % with \AtEndDocument after him and thus
                          % pertubing the last page reference, this package
                          % should be loaded last.

%-------------------------------------------------------------------------------

#include "ContextMenu.h"

void ContextMenu::createContextMenuActions()
{
    QString IconFolder = QString(MMDir().c_str()) + "data/ico/";

    _rename      = new QAction(tr("Rename"), this);
    _showNames   = new QAction(tr("Show names"), this);
    _showRefuted = new QAction(tr("Show refuted points"), this);

    _highLight = new QAction(QIcon(IconFolder + "HL.ico"),              tr("Highlight"), this);

    _AllW      = new QAction(tr("AllW") , this);
    _ThisW     = new QAction(tr("ThisW"), this);
    _ThisP     = new QAction(tr("ThisP"), this);

    _validate  = new QAction(QIcon(IconFolder + "smile.ico"),           tr("Validate"), this);
    _dubious   = new QAction(QIcon(IconFolder + "interrogation.ico"),   tr("Dubious") , this);
    _refuted   = new QAction(QIcon(IconFolder + "refuted.ico"),         tr("Refuted") , this);
    _noSaisie  = new QAction(QIcon(IconFolder + "vide.ico"),            tr("Not captured"), this);

    connect(_rename,		    SIGNAL(triggered()),   this, SLOT(rename()));
    connect(_showNames,		    SIGNAL(triggered()),   this, SLOT(showNames()));
    connect(_showRefuted,		SIGNAL(triggered()),   this, SLOT(showRefuted()));

    connect(_highLight,		    SIGNAL(triggered()),   this, SLOT(highlight()));

    _signalMapper = new QSignalMapper (this);

    /*connect(_AllW,      	    SIGNAL(triggered()),   _signalMapper, SLOT(AllW()));
    connect(_ThisW,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisW()));
    connect(_ThisP,             SIGNAL(triggered()),   _signalMapper, SLOT(ThisP()));*/

    connect(_validate,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_dubious,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_refuted,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));
    connect(_noSaisie,		    SIGNAL(triggered()),   _signalMapper, SLOT(map()));

    _signalMapper->setMapping (_validate,  NS_SaisiePts::eEPI_Valide);
    _signalMapper->setMapping (_dubious,   NS_SaisiePts::eEPI_Douteux);
    _signalMapper->setMapping (_refuted,   NS_SaisiePts::eEPI_Refute);
    _signalMapper->setMapping (_noSaisie,  NS_SaisiePts::eEPI_NonSaisi);

    connect (_signalMapper, SIGNAL(mapped(int)), this, SLOT(setPointState(int)));
}

void ContextMenu::setPointState(int state)
{
    int idx = _polygon->setNearestPointState(_lastPosImage, state);

    emit    changeState(state, idx);
}

void ContextMenu::highlight()
{
    int idx = _polygon->highlightNearestPoint(_lastPosImage);

    emit changeState(NS_SaisiePts::eEPI_Highlight, idx);
}

void ContextMenu::rename()
{
    QInputDialog* inputDialog = new QInputDialog();
    inputDialog->setOptions(QInputDialog::NoButtons);

    QString text = inputDialog->getText(NULL, tr("Rename"), tr("Point name:"), QLineEdit::Normal, _polygon->getNearestPointName(_lastPosImage));

    if (!text.isEmpty())

         _polygon->rename(_lastPosImage, text);
}

void ContextMenu::showNames()
{
    _polygon->showNames(!_polygon->bShowNames());
}

void ContextMenu::showRefuted()
{   
    _polygon->showRefuted();

    emit showRefuted(_polygon->bShowRefuted());
}

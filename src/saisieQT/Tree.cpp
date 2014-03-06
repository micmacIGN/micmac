#include "Tree.h"

TreeItem::TreeItem(const QList<QVariant> &data, TreeItem *parent)
{
    parentItem = parent;
    itemData = data;   
}

TreeItem::~TreeItem()
{
    qDeleteAll(childItems);
}

void TreeItem::appendChild(TreeItem *item)
{
    childItems.append(item);
}

TreeItem *TreeItem::child(int row)
{
    return childItems.value(row);
}

int TreeItem::childCount() const
{
    return childItems.count();
}

int TreeItem::columnCount() const
{
    return itemData.count();
}

QVariant TreeItem::data(int column) const
{
    return itemData.value(column);
}

void TreeItem::setData(const QVariant &value, int role)
{
    itemData << value;
}

TreeItem *TreeItem::parent()
{
    return parentItem;
}

int TreeItem::row() const
{
    if (parentItem)
        return parentItem->childItems.indexOf(const_cast<TreeItem*>(this));

    return 0;
}

TreeModel::TreeModel(QObject *parent):
    QAbstractItemModel(parent)
{
    QList<QVariant> rootData;
    rootData << "Point" << "Image" << "State" << "Coordinates";
    rootItem = new TreeItem(rootData);
}

TreeModel::~TreeModel()
{
    delete rootItem;
}

int TreeModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return static_cast<TreeItem*>(parent.internalPointer())->columnCount();
    else
        return rootItem->columnCount();
}

void TreeModel::setAppli(cAppli_SaisiePts *appli)
{
    _appli = appli;

    setupModelData();
}

QString StateToQString(eEtatPointeImage state)
{
    switch (state)
    {
    case eEPI_NonSaisi:
        return "Non Saisi";
    case eEPI_Refute:
        return "Refute";
    case eEPI_Douteux:
        return "Douteux";
    case eEPI_Valide:
        return "Valide";
    case eEPI_NonValue:
        return "NonValue";
    case eEPI_Disparu:
        return "Disparu";
    case eEPI_Highlight:  //only for QT interface
    case eEPI_Deleted:    //only for QT interface
        return "";
    }

    return "";
}

QVariant TreeModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (role == Qt::DisplayRole || role == Qt::EditRole)
    {
        TreeItem *item = static_cast<TreeItem*>(index.internalPointer());

        return item->data(index.column());
    }
    else
        return QVariant();
}

Qt::ItemFlags TreeModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return 0;

    return QAbstractItemModel::flags(index) | Qt::ItemIsEditable;
}

QVariant TreeModel::headerData(int section, Qt::Orientation orientation,
                               int role) const
{
    if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
        return rootItem->data(section);

    return QVariant();
}

bool TreeModel::setData(const QModelIndex &index,
                        const QVariant &value, int role)
 {
     if (index.isValid() && role == Qt::EditRole)
     {
         if (index.column() == 0) //point name
         {
             std::string oldName = index.data(Qt::DisplayRole).toString().toStdString();

             _appli->ChangeName(oldName,value.toString().toStdString());

             TreeItem *Item = static_cast<TreeItem*>(index.internalPointer());

             Item->setData(value, Qt::DisplayRole);

             emit dataChanged(index, index);

             return true;
         }
     }
     return false;
 }

QModelIndex TreeModel::index(int row, int column, const QModelIndex &parent)
            const
{
    if (!hasIndex(row, column, parent))
        return QModelIndex();

    TreeItem *parentItem;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    TreeItem *childItem = parentItem->child(row);
    if (childItem)
        return createIndex(row, column, childItem);
    else
        return QModelIndex();
}

QModelIndex TreeModel::parent(const QModelIndex &index) const
{
    if (!index.isValid())
        return QModelIndex();

    TreeItem *childItem = static_cast<TreeItem*>(index.internalPointer());
    TreeItem *parentItem = childItem->parent();

    if (parentItem == rootItem)
        return QModelIndex();

    return createIndex(parentItem->row(), 0, parentItem);
}

int TreeModel::rowCount(const QModelIndex &parent) const
{
    TreeItem *parentItem;
    if (parent.column() > 0)
        return 0;

    if (!parent.isValid())
        parentItem = rootItem;
    else
        parentItem = static_cast<TreeItem*>(parent.internalPointer());

    return parentItem->childCount();
}

QList <QVariant> TreeModel::buildRow(cSP_PointGlob* aPG)
{
    QList<QVariant> columnData;

    QString namePt  = QString(aPG->PG()->Name().c_str());
    QString coord3d = QString::number(aPG->PG()->P3D().Val().x, 'f' ,1) + " " +
                      QString::number(aPG->PG()->P3D().Val().y, 'f' ,1) + " " +
                      QString::number(aPG->PG()->P3D().Val().z, 'f' ,1);

    columnData << namePt << "" << "" << coord3d;

    return columnData;
}

QList <QVariant> TreeModel::buildChildRow(std::pair < std::string , cSP_PointeImage*> data)
{
    QList<QVariant> columnData;

    std::string     aName = data.first;
    cSP_PointeImage* aPIm = data.second;

    cOneSaisie* aSom = aPIm->Saisie();

    QString nameImg = QString(aName.c_str());

    QString state   = StateToQString(aSom->Etat());
    QString coord2d = QString::number(aSom->PtIm().x, 'f' ,1) + " " +
                      QString::number(aSom->PtIm().y, 'f' ,1);

    columnData << "" << nameImg << state << coord2d;

    return columnData;
}

int TreeModel::getColumnSize(int column, QFontMetrics fm)
{
    int colWidth = -1;
    for (int aK=0; aK < rowCount();++aK)
    {
        QModelIndex id = index(aK, column);

        QString text = data(id, Qt::DisplayRole).toString();

        int textWidth = fm.width(text);

        if (colWidth < textWidth) colWidth = textWidth;
    }

    return colWidth;
}

void TreeModel::setupModelData()
{
    QList<TreeItem*> parents;
    parents << rootItem;

    std::vector<cSP_PointGlob *> vPts = _appli->PG();

    for (int bK=0; bK < (int) vPts.size(); ++bK)
    {
        cSP_PointGlob* aPG = vPts[bK];

        TreeItem * item = new TreeItem(buildRow(aPG), rootItem);
        parents.last()->appendChild(item);

        //Pointes image

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        for(; it != map.end(); ++it)
        {  
            item->appendChild(new TreeItem(buildChildRow(*it), item));
        }
    }
}

void TreeModel::updateData()
{
    std::vector<cSP_PointGlob *> vPts = _appli->PG();

    for (int aK=0; aK < (int) vPts.size(); ++aK)
    {
        cSP_PointGlob* aPG = vPts[aK];

        QModelIndex idx = index(aK, 0);
        TreeItem * item = static_cast<TreeItem*>(idx.internalPointer());

        item->setData(buildRow(aPG));

        //Pointes image

        std::map<std::string,cSP_PointeImage *> map = aPG->getPointes();

        std::map<std::string,cSP_PointeImage *>::const_iterator it = map.begin();

        int bK = 0;
        for(; it != map.end(); ++it, ++bK)
        {
             item->child(bK)->setData(buildChildRow(*it));
        }
    }
}


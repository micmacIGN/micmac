#ifndef TREE_H
#define TREE_H

#include <QStandardItemModel>
#include <QFontMetrics>

#include "StdAfx.h"

using namespace NS_SaisiePts;

class TreeItem : public QStandardItem
{
public:
    explicit TreeItem(const QVector<QVariant> &data, TreeItem *parent = 0);
    ~TreeItem();

    void appendChild(TreeItem *child);

    TreeItem *child(int row);
    int childCount() const;
    int columnCount() const;
    int childNumber() const;
    QVariant data(int column) const;
    bool setData(int column, const QVariant &value);

    void setData(const QVector<QVariant> &data) { itemData = data; }

    int row() const;
    TreeItem *parent();

    QList<TreeItem*> getChildItems() { return childItems; }

    bool insertChildren(int position, int count, int columns);
    bool removeChildren(int position, int count);


private:
    QList<TreeItem*> childItems;
    QVector<QVariant> itemData;
    TreeItem *parentItem;
};

class TreeModel : public QAbstractItemModel
{
    Q_OBJECT

public:
    explicit TreeModel(QObject *parent = 0);
    ~TreeModel();

    QVariant        data(const QModelIndex &index, int role) const;
    bool            setData(const QModelIndex &index, const QVariant &value, int role = Qt::EditRole);
    Qt::ItemFlags   flags(const QModelIndex &index) const;
    QVariant        headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const;
    QModelIndex     index(int row, int column, const QModelIndex &parent = QModelIndex()) const;
    QModelIndex     parent(const QModelIndex &index) const;

    bool            insertRows(int position, int rows, const QModelIndex &parent = QModelIndex());
    bool            removeRows(int position, int rows, const QModelIndex &parent = QModelIndex());

    int             rowCount    (const QModelIndex &parent = QModelIndex()) const;
    int             columnCount (const QModelIndex &parent = QModelIndex()) const;

    void            setAppli(cAppli_SaisiePts* appli);

    void            setupModelData();

    void            updateData();

    QVector<QVariant> buildRow(cSP_PointGlob *aPG);

    QVector<QVariant> buildChildRow(std::pair<std::string, cSP_PointeImage *> data);

    int             getColumnSize(int column, QFontMetrics fm);

public slots:

    void            addPoint();

private:
    TreeItem*       getItem(const QModelIndex &index) const;

    TreeItem*       rootItem;

    cAppli_SaisiePts* _appli;
};

#endif // TREE_H

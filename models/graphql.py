import copy
import functools

import requests

from mmcore.gql import client as gql_client
from mmcore.gql.client import query


class GQlEntity:
    client = gql_client.GQLClient(url=gql_client.GQL_PLATFORM_URL,
                                  headers={
                                      "content-type": "application/json",
                                      "user-agent": "JS GraphQL",
                                      "X-Hasura-Role": "admin",
                                      "X-Hasura-Admin-Secret": "mysecretkey"
                                  })

    def __init__(self, body, variables=None):
        if variables is None:
            variables = dict()
        super().__init__()
        self.body = body
        self.variables = variables

    def __call__(self, **variables):
        vars = copy.deepcopy(self.variables)
        vars |= variables
        request = requests.post(self.client.url,

                                headers=self.client.headers,
                                json={
                                    "query": self.body,
                                    "variables": vars

                                }
                                )
        print(self.body, vars, request.json())
        return list(request.json()["data"].values())




InsertTypeMarks = GQlEntity("""
mutation InsertTypeMarks($x: Float, $y: Float, $z: Float, $tag: String, $uuid: uuid) {
  insert_lht_ceiling_type_marks(objects: {x: $x, y: $y, tag: $tag, z: $z}, on_conflict: {constraint: type_marks_pkey, update_columns: tag}) {
    returning {
      tag
      uuid
      x
      y
      z
    }
  }
}""")
DeleteTypeMarksByPk = GQlEntity("""
mutation DeleteTypeMarksByPk {
  delete_lht_ceiling_type_marks_by_pk(x: $x, y: $y)
}""")
InsertTypeMarkOne = GQlEntity("""
mutation InsertTypeMarkOne($tag: String = "", $x: numeric = "", $y: numeric = "", $z: numeric = "", $floor: String = "") {
  insert_lht_ceiling_type_marks_one(object: {tag: $tag, x: $x, y: $y, z: $z, floor: $floor}, on_conflict: {constraint: type_marks_pkey, update_columns: tag}) {
    tag
    uuid
  }
}""")
UpdateTypeMarksByUUID = GQlEntity("""
mutation UpdateTypeMarksByUUID($uuid: uuid = "", $x: numeric = "", $y: numeric = "", $z: numeric = "", $tag: String = "") {
  update_lht_ceiling_type_marks(_set: {uuid: $uuid, x: $x, y: $y, z: $z, tag: $tag}, where: {uuid: {_eq: $uuid}}) {
    returning {
      tag
      uuid
      x
      y
      z
    }
    affected_rows
  }
}""")
UpdateTypeMarksByPK = GQlEntity("""
mutation MyMutation7($x: numeric = "", $y: numeric = "", $tag: String = "", $uuid: uuid = "", $z: numeric = "") {
  update_lht_ceiling_type_marks_by_pk(pk_columns: {x: $x, y: $y}, _set: {tag: $tag, uuid: $uuid, z: $z}) {
    tag
    uuid
    x
    y
    z
  }
}

""")
TypeMarksNotation = {"tag": 'String = ""', "uuid": 'uuid = ""', "z": 'numeric = ""'}
PanelTriangleNotation = {
    "tag": 'String = ""',
    "uuid": 'uuid = ""',
    "z": 'numeric = ""',
    "x": 'numeric = ""',
    "y": 'numeric = ""',
    "centroid": "jsonb = {}",
    "cutted": "Boolean = false",
    "subtype": "Int = 0",
    "stripe": "Int=0",
    "points": "jsonb={}",
    "outside": "Boolean = false",
    "stage": "String = \"\"",
    "matrix": "jsonb = []",
    "mask": "Boolean = false",
    "mark": "String = \"\"",
    "floor": "String =\"\"",
    "extra": "Boolean = false"
}


def UpdateTypeMarksByPKPartialTemp(attr):
    query = """
mutation MyMutation"attr($x: numeric = "", $y: numeric = "", $$attr: $header) {
  update_lht_ceiling_type_marks_by_pk(pk_columns: {x: $x, y: $y}, _set: {$attr:$$attr}) {
    $attr
  }
}"""
    return query.replace("$attr", attr).replace("$header", TypeMarksNotation[attr])


TypeMarksByUUID = GQlEntity("""
query TypeMarksByUUID($_eq: uuid = "") {
  lht_ceiling_type_marks(where: {uuid: {_eq: $_eq}}) {
    uuid
    tag
    x
    y
    z
  }
}""")
TypeMarksByPk = GQlEntity("""
#!language=graphql
query TypeMarksByPk($x: numeric = "", $y: numeric = "") {
  lht_ceiling_type_marks_by_pk(x: $x, y: $y) {
    tag
    uuid
    x
    y
    z
  }
}""")
TypeMarksReg = GQlEntity("""
#!language=graphql
query {
  lht_ceiling_type_marks {
    tag
    uuid
    x
    y
    z
  }
}""")


def TypeMarksByPkPartial(attr): return GQlEntity("""

query TypeMarksByPk($x: numeric = "", $y: numeric = "") {
  lht_ceiling_type_marks_by_pk(x: $x, y: $y) {
    $attr
  }
}""".replace("$attr", attr))


PanelTriangleReg = GQlEntity("""
#!language=graphql
query {
  lht_ceiling_panels{
    centroid
    subtype
    tag
    cutted
    extra
    outside
    
  }
}""")

InsertTriangleOne = GQlEntity("""
mutation InsertTriangleOne($tag: String = "", $x: numeric = "", $y: numeric = "", $z: numeric = "", $floor: String = "") {
  insert_lht_ceiling_type_marks_one(object: {tag: $tag, x: $x, y: $y, z: $z, floor: $floor}, on_conflict: {constraint: type_marks_pkey, update_columns: tag}) {
    tag
    uuid
  }
}""")

InsertPanelTriangle = GQlEntity("""
mutation PanelTriangleInsert($y: numeric = 0, $x: numeric = 0, $centroid: jsonb = {}, $cutted: Boolean = false, $tag: String="", $subtype: Int = 0, $stripe: Int=0, $points: jsonb={}, $outside: Boolean = false, $stage: String = "", $matrix: jsonb = [], $mask: Boolean = false, $mark: String = "", $floor: String ="", $extra: Boolean = false) {
  insert_lht_ceiling_panels_one(object: {y: $y, x: $x, subtype: $subtype, stripe: $stripe, stage: $stage, points: $points, outside: $outside, matrix: $matrix, cutted: $cutted,mask: $mask, mark: $mark, floor: $floor, extra: $extra, centroid: $centroid, tag: $tag}, on_conflict: {constraint: panels_pkey, update_columns: tag}) {
    y
    x
    uuid
    tag
    updated_at
  }
}



""")


def PanelTriangleByPkPartial(attr): return GQlEntity("""

query PanelTriangleByPk($x: numeric = "", $y: numeric = "", $floor: String="") {
  lht_ceiling_panels_by_pk(x: $x, y: $y, floor: $floor) {
    $attr
  }
}""".replace("$attr", attr))


def UpdatePanelTriangleByPKPartialTemp(attr):
    query = """
mutation MyMutation$attr($x: numeric = "", $y: numeric = "", $floor: String = "", $$attr: $header) {
  update_lht_ceiling_panels_by_pk(pk_columns: {x: $x, y: $y, floor: $floor}, _set: {$attr:$$attr}) {
    $attr
  }
}"""
    return query.replace("$attr", attr).replace("$header", PanelTriangleNotation[attr])



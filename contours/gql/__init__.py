from mm import traverse_cxm_data_json
import json
from typing import NewType, Any
# import Rhino.Display
# Rhino.DocObjects.ObjectAttributes.PlotColorSource.

import dotenv

dotenv.load_dotenv()
import cxmdata
from mmcore.gql.client import GQLFileBasedQuery, uuid
import strawberry
from strawberry.fastapi import GraphQLRouter



insert_contour = GQLFileBasedQuery("data/insert-contour.graphql")
update_contour = GQLFileBasedQuery("data/update-contour.graphql")
delete_contour = GQLFileBasedQuery("data/delete-contour.graphql")


@strawberry.type
class Connector:
    name: str
    offset: float
    type: str | None = None
    production: str | None = None


contour_by_pk = GQLFileBasedQuery("data/query_by_pk.graphql")
connector_by_contour_pk = GQLFileBasedQuery("data/cnt_by_pk.graphql")
Curve = strawberry.scalar(
    NewType("Curve", object),
    description="The `Curve` cxmdata geometry scalar type",
    serialize=lambda x: {"cxmdata": cxmdata.CxmData.compress(x).decode()},
    parse_value=lambda x: traverse_cxm_data_json(json.loads(x))
)


@strawberry.type
class Contour:
    uuid: str

    @strawberry.field
    def detail(self) -> Connector:
        return Connector(**list(connector_by_contour_pk(
            variables={"uuid": self.uuid},

        ).values())[0]["connector"])

    @strawberry.field
    def floor(self) -> str:
        return list(contour_by_pk(variables={"uuid": self.uuid}, fields=("floor",)).values())[0]["floor"]

    @strawberry.field
    def curve(self) -> Curve:
        return Curve(
            list(contour_by_pk(variables={"uuid": self.uuid}, fields=("curve",), full_root=False).values())[0]["curve"])


query2 = GQLFileBasedQuery("data/query2.graphql")

from strawberry.type import StrawberryList
from strawberry.scalars import JSON

ContourList = StrawberryList(of_type=Contour)


@strawberry.type
class ContoursQuery:
    @strawberry.field
    def contours(self) -> ContourList:
        lst = []
        for i in list(query2().values())[0]:
            lst.append(Contour(uuid=i["uuid"]))
        return lst

    @strawberry.field
    def contour(self, uuid: str) -> Contour:
        return Contour(uuid=uuid)


@strawberry.type
class ContourMutation:
    @strawberry.field
    def insert(self, curve: JSON, detail: str, floor: str) -> JSON:
        return list(insert_contour(variables=dict(curve=curve, detail=detail, floor=floor), fields=("uuid",)).values())[
            0]

    @strawberry.field
    def update(self, curve: JSON, detail: str, floor: str) -> JSON:
        return list(update_contour(variables=dict(uuid=uuid, curve=curve, detail=detail, floor=floor),
                                   fields=("uuid",)).values())[0]

    @strawberry.field
    def delete(self, uuid: str) -> JSON:
        return list(delete_contour(variables=dict(uuid=uuid), fields=("uuid",)).values())[0]


schema = strawberry.Schema(query=ContoursQuery,mutation=ContourMutation, types=(Contour, Curve, Connector))

graphql_app = GraphQLRouter(schema)

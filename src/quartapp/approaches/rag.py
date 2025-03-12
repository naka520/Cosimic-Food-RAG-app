import json
from collections.abc import AsyncIterator

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from quartapp.approaches.base import ApproachesBase
from quartapp.approaches.schemas import DataPoint


def get_data_points(documents: list[Document]) -> list[DataPoint]:
    data_points: list[DataPoint] = []

    for res in documents:
        raw_data = json.loads(res.page_content)
        json_data_point: DataPoint = DataPoint()
        json_data_point.name = raw_data.get("name")
        json_data_point.description = raw_data.get("description")
        json_data_point.price = raw_data.get("price")
        json_data_point.category = raw_data.get("category")
        data_points.append(json_data_point)
    return data_points


REPHRASE_PROMPT = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""

CONTEXT_PROMPT = """\
あなたは勤怠と契約が辻褄が合っているか確認するアシスタントです。​
以下は例です。​

#勤怠簿の例​

2020年11月支払分（対象期間: 2024年10⽉1⽇ 〜 2024年10⽉31⽇）                    ​

山田 花子 (従業員番号: 1)                    ​

                    ​

日付    始業時間    休憩時間    終業時間    残業時間    総労働時間​

2025/3/3    9:14    0:51:00    20:11    2:30:00    0 days 10:06:00​

2025/3/4    8:45    0:48:00    20:41    3:00:00    0 days 11:08:00​

2025/3/5    9:21    0:47:00    18:50    0:30:00    0 days 08:42:00​

2025/3/6    8:58    0:58:00    19:12    1:00:00    0 days 09:16:00​

2025/3/7    8:59    1:08:00    19:52    2:00:00    0 days 09:45:00​

2025/3/10    8:48    0:50:00    18:43    0:45:00    0 days 09:05:00​

2025/3/11    9:02    1:14:00    18:21    0:30:00    0 days 08:05:00​

2025/3/12    8:35    0:51:00    18:47    0:30:00    0 days 09:21:00​

2025/3/13    9:14    0:51:00    19:46    1:30:00    0 days 09:41:00​

2025/3/14    9:17    1:09:00    19:13    1:00:00    0 days 08:47:00​

2025/3/15    9:17    1:09:00    19:13    1:00:00    1 days 08:47:00​

2025/3/17    9:11    1:03:00    20:03    2:30:00    0 days 09:49:00​

2025/3/18    8:30    0:54:00    18:33    1:00:00    0 days 09:09:00​

2025/3/19    9:26    0:57:00    18:40    1:00:00    0 days 08:17:00​

2025/3/20    9:08    1:14:00    19:45    1:30:00    0 days 09:23:00​

2025/3/21    9:03    0:54:00    19:46    2:00:00    0 days 09:49:00​

2025/3/24    8:53    1:05:00    18:05    0:00:00    0 days 08:07:00​

2025/3/25    8:56    1:01:00    18:32    0:30:00    0 days 08:35:00​

2025/3/26    9:08    1:03:00    18:36    0:15:00    0 days 08:25:00​

2025/3/27    9:23    1:15:00    18:01    0:00:00    0 days 07:23:00​

2025/3/28    8:32    0:56:00    18:51    1:00:00    0 days 09:23:00​

2025/3/31    8:58    1:12:00    18:50    0:45:00    0 days 08:40:00​

​

#雇用契約書の例​

雇 用 契 約 書​

​

株式会社ダミー（以下「甲」という。）と山田 花子（以下「乙」という。）とは、以下の条件に基づき、雇用契約を締結する。​

​

契約期間​

期間の定め有り（2024年10月1日～2025年9月30日）​

​

雇用形態​

契約社員​

​

就業の場所​

福岡オフィス​

但し、業務の都合により変更する場合がある。​

​

従事する業務内容​

事務総務​

​

所定労働時間​

​

始業・終業の時刻 ：（始業）9時00分 ～ （終業）18時00分​

休憩時間 ： 1 時間​

1 週間の所定労働時間 ： 40 時間 00 分​

但し、業務の都合により、始業・終業の時刻を変更する場合がある。​

​

時間外労働​

​

所定時間外労働の有無 ： 有​

休日労働の有無 ： 有​

​

休日​

土曜日、日曜日、祝日、GW、年末年始​

但し、休日を振替えるなどの方法によって休日を変更する場合がある。​

​

休暇​

​

年次有給休暇 所定労働日数の 8 割以上出勤し、6 ヶ月以上継続勤務した場合、法定通り​

​

賃金​

​

基本賃金（時給） ： 1,200円​

諸手当 ： 通勤手当：当社基準により支給（基本給には含まない）​

賃金締切日 ： 毎月末日​

賃金支払日 ： 翌月 25 日​

支払方法 ： 指定口座へ振込み​

給与改定 ： 1 年に 1 度​

賞与 ： 有 （業績により支給）​

退職金 ： 無​

​

退職に関する事項​

自己都合退職を希望する場合、少なくとも 30 日以上前に届け出ること​

​

更新の有無​

更新する場合があり得る​

​

判断の基準​

従事している業務の進捗状況により判断する​

​

社会保険等の加入​

労災保険、雇用保険、健康保険、厚生年金​

​

備考​

本契約に定めのない事項については、就業規則の定めによる。​

​

本契約の締結の証として、本書 2 通を作成し、甲乙それぞれ 1 通を所持するものとする。​

​

2024年10月1日​

​

（甲） 名 称：Help Tech 株式会社​

所在地：福岡県福岡市中央区薬院4-8-28-203​

代表者：代表取締役 高橋ダミー 印​

​

（乙） 住 所：​

氏 名：山田 花子 印​\

User Question: {input}

Chatbot Response:"""


class RAG(ApproachesBase):
    async def run(
        self, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> tuple[list[Document], str]:
        # Create a vector store retriever
        retriever = self._vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": limit, "score_threshold": score_threshold}
        )

        self._chat.temperature = 0.3

        # Create a vector context aware chat retriever
        rephrase_prompt_template = ChatPromptTemplate.from_template(REPHRASE_PROMPT)
        rephrase_chain = rephrase_prompt_template | self._chat

        # Rephrase the question
        rephrased_question = await rephrase_chain.ainvoke({"chat_history": messages[:-1], "question": messages[-1]})

        print(rephrased_question.content)
        # Perform vector search
        vector_context = await retriever.ainvoke(str(rephrased_question.content))
        data_points: list[DataPoint] = get_data_points(vector_context)

        # Create a vector context aware chat retriever
        context_prompt_template = ChatPromptTemplate.from_template(CONTEXT_PROMPT)
        self._chat.temperature = temperature
        context_chain = context_prompt_template | self._chat
        documents_list: list[Document] = []
        if data_points:
            # Perform RAG search
            response = await context_chain.ainvoke(
                {"context": [dp.to_dict() for dp in data_points], "input": rephrased_question.content}
            )
            for document in vector_context:
                documents_list.append(
                    Document(page_content=document.page_content, metadata={"source": document.metadata["source"]})
                )
            formatted_response = (
                f'{{"response": "{response.content}", "rephrased_response": "{rephrased_question.content}"}}'
            )
            return documents_list, str(formatted_response)

        # Perform RAG search with no context
        response = await context_chain.ainvoke({"context": [], "input": rephrased_question.content})
        return [], str(response.content)

    async def run_stream(
        self, messages: list, temperature: float, limit: int, score_threshold: float
    ) -> tuple[list[Document], AsyncIterator[BaseMessage]]:
        # Create a vector store retriever
        retriever = self._vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": limit, "score_threshold": score_threshold}
        )

        self._chat.temperature = 0.3

        # Create a vector context aware chat retriever
        rephrase_prompt_template = ChatPromptTemplate.from_template(REPHRASE_PROMPT)
        rephrase_chain = rephrase_prompt_template | self._chat

        # Rephrase the question
        rephrased_question = await rephrase_chain.ainvoke({"chat_history": messages[:-1], "question": messages[-1]})

        print(rephrased_question.content)
        # Perform vector search
        vector_context = await retriever.ainvoke(str(rephrased_question.content))
        data_points: list[DataPoint] = get_data_points(vector_context)

        # Create a vector context aware chat retriever
        context_prompt_template = ChatPromptTemplate.from_template(CONTEXT_PROMPT)
        self._chat.temperature = temperature
        context_chain = context_prompt_template | self._chat
        documents_list: list[Document] = []

        if data_points:
            # Perform RAG search
            response = context_chain.astream(
                {"context": [dp.to_dict() for dp in data_points], "input": rephrased_question.content}
            )
            for document in vector_context:
                documents_list.append(
                    Document(page_content=document.page_content, metadata={"source": document.metadata["source"]})
                )
            return documents_list, response

        # Perform RAG search with no context
        response = context_chain.astream({"context": [], "input": rephrased_question.content})
        return [], response

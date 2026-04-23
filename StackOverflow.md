---
jupyter:
  colab:
    collapsed_sections:
    - VJrUUVeWmecT
    - vJcnv-IPYaCc
    - 87fb81eb-3c25-4255-bcc3-0fdc5180c3f6
    - SsFj2SK-mqBC
    - yGNOxz8Sql6H
    - YSlxPvF9yh07
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.12.3
  nbformat: 4
  nbformat_minor: 5
---

::: {#efde5a15-8fbd-49af-8118-4492aa7b3bc3 .cell .code execution_count="73" id="efde5a15-8fbd-49af-8118-4492aa7b3bc3"}
``` python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
```
:::

::: {#VJrUUVeWmecT .cell .markdown id="VJrUUVeWmecT"}
# Data Cleaning
:::

::: {#3a1f39d1-8ef6-43dc-9e31-680976e62c36 .cell .code execution_count="74" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="3a1f39d1-8ef6-43dc-9e31-680976e62c36" outputId="fb2b67e3-6fc9-47fc-ebe1-033ede9d7ad5"}
``` python
data1 = pd.read_csv('survey_results_public.csv')#,on_bad_lines='skip')
```

::: {.output .stream .stderr}
    /tmp/ipykernel_926/274616350.py:1: DtypeWarning: Columns (56,74,92,97,98,105,109,110,132,162,165) have mixed types. Specify dtype option on import or set low_memory=False.
      data1 = pd.read_csv('survey_results_public.csv')#,on_bad_lines='skip')
:::
:::

::: {#0Z5XAA2nmch9 .cell .markdown id="0Z5XAA2nmch9"}
#### choosing columns for modeling
:::

::: {#8096b6e4-5ad1-4a4a-923e-895e0b1c0541 .cell .code execution_count="75" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8096b6e4-5ad1-4a4a-923e-895e0b1c0541" outputId="99aa8cba-8ce8-41d6-86a9-3ec12c60cf6d"}
``` python
print(list(data1.columns))
```

::: {.output .stream .stdout}
    ['ResponseId', 'MainBranch', 'Age', 'EdLevel', 'Employment', 'EmploymentAddl', 'WorkExp', 'LearnCodeChoose', 'LearnCode', 'LearnCodeAI', 'AILearnHow', 'YearsCode', 'DevType', 'OrgSize', 'ICorPM', 'RemoteWork', 'PurchaseInfluence', 'TechEndorseIntro', 'TechEndorse_1', 'TechEndorse_2', 'TechEndorse_3', 'TechEndorse_4', 'TechEndorse_5', 'TechEndorse_6', 'TechEndorse_7', 'TechEndorse_8', 'TechEndorse_9', 'TechEndorse_13', 'TechEndorse_13_TEXT', 'TechOppose_1', 'TechOppose_2', 'TechOppose_3', 'TechOppose_5', 'TechOppose_7', 'TechOppose_9', 'TechOppose_11', 'TechOppose_13', 'TechOppose_16', 'TechOppose_15', 'TechOppose_15_TEXT', 'Industry', 'JobSatPoints_1', 'JobSatPoints_2', 'JobSatPoints_3', 'JobSatPoints_4', 'JobSatPoints_5', 'JobSatPoints_6', 'JobSatPoints_7', 'JobSatPoints_8', 'JobSatPoints_9', 'JobSatPoints_10', 'JobSatPoints_11', 'JobSatPoints_13', 'JobSatPoints_14', 'JobSatPoints_15', 'JobSatPoints_16', 'JobSatPoints_15_TEXT', 'AIThreat', 'NewRole', 'ToolCountWork', 'ToolCountPersonal', 'Country', 'Currency', 'CompTotal', 'LanguageChoice', 'LanguageHaveWorkedWith', 'LanguageWantToWorkWith', 'LanguageAdmired', 'LanguagesHaveEntry', 'LanguagesWantEntry', 'DatabaseChoice', 'DatabaseHaveWorkedWith', 'DatabaseWantToWorkWith', 'DatabaseAdmired', 'DatabaseHaveEntry', 'DatabaseWantEntry', 'PlatformChoice', 'PlatformHaveWorkedWith', 'PlatformWantToWorkWith', 'PlatformAdmired', 'PlatformHaveEntry', 'PlatformWantEntry', 'WebframeChoice', 'WebframeHaveWorkedWith', 'WebframeWantToWorkWith', 'WebframeAdmired', 'WebframeHaveEntry', 'WebframeWantEntry', 'DevEnvsChoice', 'DevEnvsHaveWorkedWith', 'DevEnvsWantToWorkWith', 'DevEnvsAdmired', 'DevEnvHaveEntry', 'DevEnvWantEntry', 'SOTagsHaveWorkedWith', 'SOTagsWantToWorkWith', 'SOTagsAdmired', 'SOTagsHaveEntry', 'SOTagsWant Entry', 'OpSysPersonal use', 'OpSysProfessional use', 'OfficeStackAsyncHaveWorkedWith', 'OfficeStackAsyncWantToWorkWith', 'OfficeStackAsyncAdmired', 'OfficeStackHaveEntry', 'OfficeStackWantEntry', 'CommPlatformHaveWorkedWith', 'CommPlatformWantToWorkWith', 'CommPlatformAdmired', 'CommPlatformHaveEntr', 'CommPlatformWantEntr', 'AIModelsChoice', 'AIModelsHaveWorkedWith', 'AIModelsWantToWorkWith', 'AIModelsAdmired', 'AIModelsHaveEntry', 'AIModelsWantEntry', 'SOAccount', 'SOVisitFreq', 'SODuration', 'SOPartFreq', 'SO_Dev_Content', 'SO_Actions_1', 'SO_Actions_16', 'SO_Actions_3', 'SO_Actions_4', 'SO_Actions_5', 'SO_Actions_6', 'SO_Actions_9', 'SO_Actions_7', 'SO_Actions_10', 'SO_Actions_15', 'SO_Actions_15_TEXT', 'SOComm', 'SOFriction', 'AISelect', 'AISent', 'AIAcc', 'AIComplex', 'AIToolCurrently partially AI', "AIToolDon't plan to use AI for this task", 'AIToolPlan to partially use AI', 'AIToolPlan to mostly use AI', 'AIToolCurrently mostly AI', 'AIFrustration', 'AIExplain', 'AIAgents', 'AIAgentChange', 'AIAgent_Uses', 'AgentUsesGeneral', 'AIAgentImpactSomewhat agree', 'AIAgentImpactNeutral', 'AIAgentImpactSomewhat disagree', 'AIAgentImpactStrongly agree', 'AIAgentImpactStrongly disagree', 'AIAgentChallengesNeutral', 'AIAgentChallengesSomewhat disagree', 'AIAgentChallengesStrongly agree', 'AIAgentChallengesSomewhat agree', 'AIAgentChallengesStrongly disagree', 'AIAgentKnowledge', 'AIAgentKnowWrite', 'AIAgentOrchestration', 'AIAgentOrchWrite', 'AIAgentObserveSecure', 'AIAgentObsWrite', 'AIAgentExternal', 'AIAgentExtWrite', 'AIHuman', 'AIOpen', 'ConvertedCompYearly', 'JobSat']
:::
:::

::: {#32c170ec-69e6-409f-8118-bdeace44d61b .cell .code execution_count="76" id="32c170ec-69e6-409f-8118-bdeace44d61b"}
``` python
org_data = pd.DataFrame(data1, columns=['MainBranch','Age','EdLevel','Employment','WorkExp','YearsCode','DevType','OrgSize','RemoteWork','Industry','Country','LanguageHaveWorkedWith',
                                   'DatabaseHaveWorkedWith','PlatformHaveWorkedWith','WebframeHaveWorkedWith','DevEnvsHaveWorkedWith',
                                    'AIModelsHaveWorkedWith','ConvertedCompYearly'])
```
:::

::: {#X8pL9grhiXvi .cell .markdown id="X8pL9grhiXvi"}
#### removing duplicates
:::

::: {#e2af37e3-ba50-4886-bf9c-263124bc234f .cell .code execution_count="77" id="e2af37e3-ba50-4886-bf9c-263124bc234f"}
``` python
org_data = org_data.drop_duplicates()
```
:::

::: {#b1911f85-7f6e-4719-8f6b-ef1b2f8879f7 .cell .code execution_count="78" id="b1911f85-7f6e-4719-8f6b-ef1b2f8879f7"}
``` python
org_data = org_data[org_data['ConvertedCompYearly'].notna()]
```
:::

::: {#800d3e65-c1d5-4be1-b5c2-728367b7c199 .cell .markdown id="800d3e65-c1d5-4be1-b5c2-728367b7c199"}
#### dropping columns with more than 50% missing values
:::

::: {#E1l-OrjNiOes .cell .code execution_count="79" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":648}" id="E1l-OrjNiOes" outputId="3b5698dc-f674-404e-98c6-f672c53bb047"}
``` python
org_data.isnull().sum()*100/(len(org_data))
```

::: {.output .execute_result execution_count="79"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MainBranch</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>EdLevel</th>
      <td>0.070996</td>
    </tr>
    <tr>
      <th>Employment</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>WorkExp</th>
      <td>1.996241</td>
    </tr>
    <tr>
      <th>YearsCode</th>
      <td>0.455210</td>
    </tr>
    <tr>
      <th>DevType</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>OrgSize</th>
      <td>11.451242</td>
    </tr>
    <tr>
      <th>RemoteWork</th>
      <td>11.902276</td>
    </tr>
    <tr>
      <th>Industry</th>
      <td>3.817081</td>
    </tr>
    <tr>
      <th>Country</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>LanguageHaveWorkedWith</th>
      <td>7.625809</td>
    </tr>
    <tr>
      <th>DatabaseHaveWorkedWith</th>
      <td>22.848194</td>
    </tr>
    <tr>
      <th>PlatformHaveWorkedWith</th>
      <td>25.387346</td>
    </tr>
    <tr>
      <th>WebframeHaveWorkedWith</th>
      <td>30.661934</td>
    </tr>
    <tr>
      <th>DevEnvsHaveWorkedWith</th>
      <td>21.482564</td>
    </tr>
    <tr>
      <th>AIModelsHaveWorkedWith</th>
      <td>49.914387</td>
    </tr>
    <tr>
      <th>ConvertedCompYearly</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>
```
:::
:::

::: {#96801167-04ff-42e9-b587-91cad973cb7b .cell .code execution_count="80" id="96801167-04ff-42e9-b587-91cad973cb7b"}
``` python
org_data = org_data.drop(columns='AIModelsHaveWorkedWith',axis=1)
```
:::

::: {#d33a63f1-e3ae-4394-8000-1247f66aa2ed .cell .code execution_count="81" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="d33a63f1-e3ae-4394-8000-1247f66aa2ed" outputId="7e50ee66-e8a4-4a60-f67f-98ec5145ee3e"}
``` python
org_data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 23945 entries, 0 to 49122
    Data columns (total 17 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   MainBranch              23945 non-null  object 
     1   Age                     23945 non-null  object 
     2   EdLevel                 23928 non-null  object 
     3   Employment              23945 non-null  object 
     4   WorkExp                 23467 non-null  float64
     5   YearsCode               23836 non-null  float64
     6   DevType                 23945 non-null  object 
     7   OrgSize                 21203 non-null  object 
     8   RemoteWork              21095 non-null  object 
     9   Industry                23031 non-null  object 
     10  Country                 23945 non-null  object 
     11  LanguageHaveWorkedWith  22119 non-null  object 
     12  DatabaseHaveWorkedWith  18474 non-null  object 
     13  PlatformHaveWorkedWith  17866 non-null  object 
     14  WebframeHaveWorkedWith  16603 non-null  object 
     15  DevEnvsHaveWorkedWith   18801 non-null  object 
     16  ConvertedCompYearly     23945 non-null  float64
    dtypes: float64(3), object(14)
    memory usage: 3.3+ MB
:::
:::

::: {#d38b7acf-a12d-46c8-9b09-ac5287716e3e .cell .code execution_count="82" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":642}" id="d38b7acf-a12d-46c8-9b09-ac5287716e3e" outputId="05909d64-c68d-41d3-b600-2b4dedf522b4"}
``` python
org_data.head()
```

::: {.output .execute_result execution_count="82"}
``` json
{"summary":"{\n  \"name\": \"org_data\",\n  \"rows\": 23945,\n  \"fields\": [\n    {\n      \"column\": \"MainBranch\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"I am a developer by profession\",\n          \"I am not primarily a developer, but I write code sometimes as part of my work/studies\",\n          \"I am learning to code\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"25-34 years old\",\n          \"35-44 years old\",\n          \"55-64 years old\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"EdLevel\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"Associate degree (A.A., A.S., etc.)\",\n          \"Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)\",\n          \"Master\\u2019s degree (M.A., M.S., M.Eng., MBA, etc.)\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Employment\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"Employed\",\n          \"Independent contractor, freelancer, or self-employed\",\n          \"I prefer not to say\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WorkExp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10.130644595058582,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 66,\n        \"samples\": [\n          60.0,\n          61.0,\n          8.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"YearsCode\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11.062438270261678,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 66,\n        \"samples\": [\n          57.0,\n          59.0,\n          14.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevType\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 32,\n        \"samples\": [\n          \"Developer, QA or test\",\n          \"Developer, embedded applications or devices\",\n          \"Retired\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"OrgSize\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 9,\n        \"samples\": [\n          \"Just me - I am a freelancer, sole proprietor, etc.\",\n          \"500 to 999 employees\",\n          \"100 to 499 employees\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RemoteWork\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Hybrid (some in-person, leans heavy to flexibility)\",\n          \"In-person\",\n          \"Hybrid (some remote, leans heavy to in-person)\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Industry\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 15,\n        \"samples\": [\n          \"Internet, Telecomm or Information Services\",\n          \"Other:\",\n          \"Fintech\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 164,\n        \"samples\": [\n          \"Kosovo\",\n          \"Cameroon\",\n          \"Bhutan\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"LanguageHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 11164,\n        \"samples\": [\n          \"Bash/Shell (all shells);F#;Go;JavaScript;SQL;TypeScript\",\n          \"Bash/Shell (all shells);Go;Groovy;HTML/CSS;Java;JavaScript;Python;Rust;SQL\",\n          \"C++;Go;HTML/CSS;JavaScript;Kotlin;Lua;PowerShell;Python;Rust;TypeScript\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DatabaseHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5433,\n        \"samples\": [\n          \"Dynamodb;Elasticsearch;MongoDB;Oracle;Snowflake\",\n          \"DuckDB;Microsoft SQL Server;Neo4J;PostgreSQL;Redis;Snowflake\",\n          \"Elasticsearch;Oracle;PostgreSQL;Redis\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"PlatformHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 14319,\n        \"samples\": [\n          \"Composer;Docker;Firebase;Google Cloud;Netlify;npm;NuGet;Pip;Supabase;Vercel;Vite;Webpack;Yarn\",\n          \"Amazon Web Services (AWS);Cloudflare;Composer;Docker;Homebrew;npm;pnpm;Vite;Webpack\",\n          \"Amazon Web Services (AWS);Kubernetes;Pip;Poetry\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WebframeHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5838,\n        \"samples\": [\n          \"Drupal;jQuery;Laravel;Symfony\",\n          \"Express;Next.js;Node.js;Phoenix;React;Ruby on Rails;Svelte\",\n          \"Ruby on Rails;Vue.js\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevEnvsHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5991,\n        \"samples\": [\n          \"Claude Code;Cursor;Visual Studio Code;Windsurf\",\n          \"PyCharm;Sublime Text;Vim;Visual Studio Code;Windsurf\",\n          \"Cursor;Neovim;Vim;Visual Studio;Visual Studio Code;Windsurf\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ConvertedCompYearly\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 461775.99983743153,\n        \"min\": 1.0,\n        \"max\": 50000000.0,\n        \"num_unique_values\": 6237,\n        \"samples\": [\n          73536.0,\n          141352.0,\n          3350.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"org_data"}
```
:::
:::

::: {#9e1891b3-00d2-450d-b2bf-7cc95a67f631 .cell .code execution_count="83" id="9e1891b3-00d2-450d-b2bf-7cc95a67f631"}
``` python
# filling null values with median or mode
for col in ['WorkExp','YearsCode']:
    org_data[col] = org_data[col].fillna(org_data[col].median())

for col in ['EdLevel','OrgSize','RemoteWork','Industry']:
    org_data[col]= org_data[col].fillna(org_data[col].mode()[0])

# filling null values and splitting strings in a column into a list
listy = ['DevType','LanguageHaveWorkedWith','DatabaseHaveWorkedWith','PlatformHaveWorkedWith','WebframeHaveWorkedWith','DevEnvsHaveWorkedWith']

for col in listy:
    org_data[col] = org_data[col].fillna('Unknown')
    org_data[col] = org_data[col].str.split(';')
```
:::

::: {#b337a61d-c102-419a-b50c-3f9b0cef63c6 .cell .code execution_count="84" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":642}" id="b337a61d-c102-419a-b50c-3f9b0cef63c6" outputId="a56be599-a001-490e-d441-9f7f4b50185b"}
``` python
org_data.head()
```

::: {.output .execute_result execution_count="84"}
``` json
{"summary":"{\n  \"name\": \"org_data\",\n  \"rows\": 23945,\n  \"fields\": [\n    {\n      \"column\": \"MainBranch\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"I am a developer by profession\",\n          \"I am not primarily a developer, but I write code sometimes as part of my work/studies\",\n          \"I am learning to code\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 7,\n        \"samples\": [\n          \"25-34 years old\",\n          \"35-44 years old\",\n          \"55-64 years old\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"EdLevel\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"Associate degree (A.A., A.S., etc.)\",\n          \"Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)\",\n          \"Master\\u2019s degree (M.A., M.S., M.Eng., MBA, etc.)\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Employment\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"Employed\",\n          \"Independent contractor, freelancer, or self-employed\",\n          \"I prefer not to say\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WorkExp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10.036107735976792,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 66,\n        \"samples\": [\n          60.0,\n          61.0,\n          8.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"YearsCode\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11.038731661692076,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 66,\n        \"samples\": [\n          57.0,\n          59.0,\n          14.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevType\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"OrgSize\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 9,\n        \"samples\": [\n          \"Just me - I am a freelancer, sole proprietor, etc.\",\n          \"500 to 999 employees\",\n          \"100 to 499 employees\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RemoteWork\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 5,\n        \"samples\": [\n          \"Hybrid (some in-person, leans heavy to flexibility)\",\n          \"In-person\",\n          \"Hybrid (some remote, leans heavy to in-person)\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Industry\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 15,\n        \"samples\": [\n          \"Internet, Telecomm or Information Services\",\n          \"Other:\",\n          \"Fintech\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 164,\n        \"samples\": [\n          \"Kosovo\",\n          \"Cameroon\",\n          \"Bhutan\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"LanguageHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DatabaseHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"PlatformHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WebframeHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevEnvsHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ConvertedCompYearly\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 461775.99983743153,\n        \"min\": 1.0,\n        \"max\": 50000000.0,\n        \"num_unique_values\": 6237,\n        \"samples\": [\n          73536.0,\n          141352.0,\n          3350.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"org_data"}
```
:::
:::

::: {#ab81d71e-7507-4401-9866-fcbdd99820a7 .cell .code execution_count="85" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="ab81d71e-7507-4401-9866-fcbdd99820a7" outputId="38664f2e-f434-4b28-f8a6-c4beb76ce890"}
``` python
# printing info about each column - number of unique values
for col in org_data.columns:
    if col in listy:
        uni = org_data[col].explode().unique().size
    else:
        uni = org_data[col].unique().size
    print(f"\nName of the column: {col}")
    print(f"Number of unique values: {uni}")
       # print(data[col].explode().value_counts())
    if uni <10:
        print(org_data[col].value_counts())
```

::: {.output .stream .stdout}

    Name of the column: MainBranch
    Number of unique values: 6
    MainBranch
    I am a developer by profession                                                                20193
    I am not primarily a developer, but I write code sometimes as part of my work/studies          2187
    I used to be a developer by profession, but no longer am                                        498
    I work with developers or my work supports developers but am not a developer by profession      390
    I am learning to code                                                                           363
    I code primarily as a hobby                                                                     314
    Name: count, dtype: int64

    Name of the column: Age
    Number of unique values: 7
    Age
    25-34 years old      8598
    35-44 years old      7581
    45-54 years old      3428
    18-24 years old      2601
    55-64 years old      1387
    65 years or older     328
    Prefer not to say      22
    Name: count, dtype: int64

    Name of the column: EdLevel
    Number of unique values: 8
    EdLevel
    Bachelor’s degree (B.A., B.S., B.Eng., etc.)                                          10442
    Master’s degree (M.A., M.S., M.Eng., MBA, etc.)                                        6917
    Some college/university study without earning a degree                                 2862
    Professional degree (JD, MD, Ph.D, Ed.D, etc.)                                         1363
    Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)     1214
    Associate degree (A.A., A.S., etc.)                                                     798
    Other (please specify):                                                                 222
    Primary/elementary school                                                               127
    Name: count, dtype: int64

    Name of the column: Employment
    Number of unique values: 6
    Employment
    Employed                                                19499
    Independent contractor, freelancer, or self-employed     3049
    Student                                                   703
    Not employed                                              504
    Retired                                                   133
    I prefer not to say                                        57
    Name: count, dtype: int64

    Name of the column: WorkExp
    Number of unique values: 66

    Name of the column: YearsCode
    Number of unique values: 66

    Name of the column: DevType
    Number of unique values: 32

    Name of the column: OrgSize
    Number of unique values: 9
    OrgSize
    20 to 99 employees                                    7031
    100 to 499 employees                                  4022
    Less than 20 employees                                3312
    10,000 or more employees                              3189
    1,000 to 4,999 employees                              2745
    500 to 999 employees                                  1574
    5,000 to 9,999 employees                              1057
    Just me - I am a freelancer, sole proprietor, etc.     704
    I don’t know                                           311
    Name: count, dtype: int64

    Name of the column: RemoteWork
    Number of unique values: 5
    RemoteWork
    Remote                                                                          9976
    Hybrid (some remote, leans heavy to in-person)                                  4202
    Hybrid (some in-person, leans heavy to flexibility)                             3815
    In-person                                                                       3199
    Your choice (very flexible, you can come in when you want or just as needed)    2753
    Name: count, dtype: int64

    Name of the column: Industry
    Number of unique values: 15

    Name of the column: Country
    Number of unique values: 164

    Name of the column: LanguageHaveWorkedWith
    Number of unique values: 43

    Name of the column: DatabaseHaveWorkedWith
    Number of unique values: 31

    Name of the column: PlatformHaveWorkedWith
    Number of unique values: 43

    Name of the column: WebframeHaveWorkedWith
    Number of unique values: 29

    Name of the column: DevEnvsHaveWorkedWith
    Number of unique values: 28

    Name of the column: ConvertedCompYearly
    Number of unique values: 6237
:::
:::

::: {#vJcnv-IPYaCc .cell .markdown id="vJcnv-IPYaCc"}
# Feature Engineering
:::

::: {#AMrGeI4njICo .cell .markdown id="AMrGeI4njICo"}
#### reducing cardinality
:::

::: {#2f7be4c7-2e66-48ee-bd0c-254edaba209d .cell .code execution_count="86" id="2f7be4c7-2e66-48ee-bd0c-254edaba209d"}
``` python
org_data['RemoteWork'] = org_data['RemoteWork'].replace(['Hybrid (some remote, leans heavy to in-person)','Hybrid (some in-person, leans heavy to flexibility)'],'Hybrid')
org_data=  org_data[org_data['RemoteWork'].isin(['Hybrid','In-person','Remote'])]
org_data = org_data[org_data['MainBranch']=='I am a developer by profession']
org_data = org_data[org_data['Employment'].isin(['Employed','Independent contractor, freelancer, or self-employed','Student'])]
org_data = org_data[org_data['Age']!= 'Prefer not to say']
org_data = org_data[org_data['OrgSize']!= 'I don’t know']
org_data['OrgSize'] = org_data['OrgSize'].replace('Just me - I am a freelancer, sole proprietor, etc.','Just me')
```
:::

::: {#1a5db8fd-b242-4436-aa77-76bf7a071d33 .cell .code execution_count="87" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":625}" id="1a5db8fd-b242-4436-aa77-76bf7a071d33" outputId="add366d5-18b1-4bfe-ef0d-7188e441c5ff"}
``` python
org_data.head()
```

::: {.output .execute_result execution_count="87"}
``` json
{"summary":"{\n  \"name\": \"org_data\",\n  \"rows\": 17265,\n  \"fields\": [\n    {\n      \"column\": \"MainBranch\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 1,\n        \"samples\": [\n          \"I am a developer by profession\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"25-34 years old\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"EdLevel\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"Associate degree (A.A., A.S., etc.)\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Employment\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Employed\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WorkExp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.55596823195557,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 59,\n        \"samples\": [\n          8.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"YearsCode\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10.598578216983547,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 62,\n        \"samples\": [\n          48.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevType\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"OrgSize\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"500 to 999 employees\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RemoteWork\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 3,\n        \"samples\": [\n          \"Remote\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Industry\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 15,\n        \"samples\": [\n          \"Internet, Telecomm or Information Services\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 156,\n        \"samples\": [\n          \"Cambodia\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"LanguageHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DatabaseHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"PlatformHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WebframeHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevEnvsHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ConvertedCompYearly\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 330712.6684792635,\n        \"min\": 1.0,\n        \"max\": 33552715.0,\n        \"num_unique_values\": 5002,\n        \"samples\": [\n          70305.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"org_data"}
```
:::
:::

::: {#8FOD8sqvlpdW .cell .markdown id="8FOD8sqvlpdW"}
#### one-hot encoding
:::

::: {#abca9af9-b9b1-4053-8383-af91ea1c65cc .cell .code execution_count="88" id="abca9af9-b9b1-4053-8383-af91ea1c65cc"}
``` python
age_order = {
    '25-34 years old':0,
    '35-44 years old':1,
    '45-54 years old':2,
    '18-24 years old':3,
    '55-64 years old':4,
    '65 years or older':5
}
orgsize_order = {
    'Just me':0,
    'Less than 20 employees':1,
    '20 to 99 employees':2,
    '100 to 499 employees':3,
    '500 to 999 employees':4,
    '1,000 to 4,999 employees':5,
    '5,000 to 9,999 employees':6,
    '10,000 or more employees':7
}

edLevel_order = {
    'Primary/elementary school':0,
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':1,
    'Some college/university study without earning a degree':2,
    'Associate degree (A.A., A.S., etc.)':3,
    'Bachelor’s degree (B.A., B.S., B.Eng., etc.)':4,
    'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)':5,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)':6,
    'Other (please specify):':2
}

employment_order = {
    'Student':0,
    'Employed':1,
    'Independent contractor, freelancer, or self-employed':2
}

remote_order = {
    'In-person':0,
    'Hybrid':1,
    'Remote':2
}
```
:::

::: {#d7a82dbc-68bd-4ed4-9f43-144cbc88b927 .cell .code execution_count="89" id="d7a82dbc-68bd-4ed4-9f43-144cbc88b927"}
``` python
data = org_data.copy()
data['Age'] = data['Age'].map(age_order)
data['OrgSize'] = data['OrgSize'].map(orgsize_order)
data['EdLevel'] = data['EdLevel'].map(edLevel_order)
data['Employment'] = data['Employment'].map(employment_order)
data['RemoteWork'] = data['RemoteWork'].map(remote_order)
```
:::

::: {#e1b79db2-3261-497f-844b-25ff8c91fe1b .cell .code execution_count="90" id="e1b79db2-3261-497f-844b-25ff8c91fe1b"}
``` python
# taking top 10 most common values from each column that has a lot of values
topCountries = data['Country'].value_counts().head(10).index
topIndustries = data['Industry'].value_counts().head(10).index

topLanguages = data['LanguageHaveWorkedWith'].explode().value_counts().head(10).index
topDatabases = data['DatabaseHaveWorkedWith'].explode().value_counts().head(10).index
topPlatforms = data['PlatformHaveWorkedWith'].explode().value_counts().head(10).index
topWebframes = data['WebframeHaveWorkedWith'].explode().value_counts().head(10).index
topDevEnvs = data['DevEnvsHaveWorkedWith'].explode().value_counts().head(10).index
topDevTypes = data['DevType'].explode().value_counts().head(10).index
```
:::

::: {#c34834d6-5847-4103-bb93-f5bdcd6e77ec .cell .code execution_count="91" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":486}" id="c34834d6-5847-4103-bb93-f5bdcd6e77ec" outputId="9c9f4144-d66f-4dc3-e825-37a9e0263621"}
``` python
data.head()
```

::: {.output .execute_result execution_count="91"}
``` json
{"summary":"{\n  \"name\": \"data\",\n  \"rows\": 17265,\n  \"fields\": [\n    {\n      \"column\": \"MainBranch\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 1,\n        \"samples\": [\n          \"I am a developer by profession\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Age\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 5,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"EdLevel\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Employment\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WorkExp\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.55596823195557,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 59,\n        \"samples\": [\n          8.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"YearsCode\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 10.598578216983547,\n        \"min\": 1.0,\n        \"max\": 100.0,\n        \"num_unique_values\": 62,\n        \"samples\": [\n          48.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevType\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"OrgSize\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 2,\n        \"min\": 0,\n        \"max\": 7,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          4\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RemoteWork\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 2,\n        \"num_unique_values\": 3,\n        \"samples\": [\n          2\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Industry\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 15,\n        \"samples\": [\n          \"Internet, Telecomm or Information Services\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Country\",\n      \"properties\": {\n        \"dtype\": \"category\",\n        \"num_unique_values\": 156,\n        \"samples\": [\n          \"Cambodia\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"LanguageHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DatabaseHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"PlatformHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"WebframeHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"DevEnvsHaveWorkedWith\",\n      \"properties\": {\n        \"dtype\": \"object\",\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ConvertedCompYearly\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 330712.6684792635,\n        \"min\": 1.0,\n        \"max\": 33552715.0,\n        \"num_unique_values\": 5002,\n        \"samples\": [\n          70305.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"data"}
```
:::
:::

::: {#85404384-a043-4487-a0e0-8c5fb18bee64 .cell .code execution_count="92" id="85404384-a043-4487-a0e0-8c5fb18bee64"}
``` python
# transforming top values into separate 1/0 columns
temp = ['LanguageHaveWorkedWith','DatabaseHaveWorkedWith','PlatformHaveWorkedWith','WebframeHaveWorkedWith','DevEnvsHaveWorkedWith','DevType']
top_list = list(topLanguages) + list(topDatabases) + list(topPlatforms) + list(topWebframes) + list(topDevEnvs) + list(topDevTypes)
mapping_top = {
    'LanguageHaveWorkedWith':topLanguages,
    'DatabaseHaveWorkedWith': topDatabases,
    'PlatformHaveWorkedWith': topPlatforms,
    'WebframeHaveWorkedWith':topWebframes,
    'DevEnvsHaveWorkedWith':topDevEnvs,
    'DevType': topDevTypes
}

for col,items in mapping_top.items():
    rep = col.replace('HaveWorkedWith','')
    for item in items:
        new_name = f'{rep}_{item}'
        data[new_name] = data[col].str.contains(item, na=False, regex=False).astype(int)
```
:::

::: {#84274ee4-bdee-46f5-9af0-8d3baf468892 .cell .code execution_count="93" id="84274ee4-bdee-46f5-9af0-8d3baf468892"}
``` python
# removing old columns
data = data.drop(temp, axis=1)
```
:::

::: {#df89fa60-110b-453f-a993-9f33f0470fc4 .cell .code execution_count="94" id="df89fa60-110b-453f-a993-9f33f0470fc4"}
``` python
# transforming top values into separate columns, other values go to the column 'Other'
data['Country_New']= data['Country'].apply(lambda x: x if x in topCountries else 'Other')
data['Industries_New']= data['Industry'].apply(lambda x: x if x in topIndustries else 'Other')

country_dummies = pd.get_dummies(data['Country_New'], prefix='Country', drop_first=True)
data = pd.concat([data,country_dummies],axis=1)

ind_dummies = pd.get_dummies(data['Industries_New'], prefix='Industry', drop_first=True)
data = pd.concat([data,ind_dummies],axis=1)

# removing old columns
data = data.drop(['Country','Country_New','Industry','Industries_New','MainBranch'],axis=1)
```
:::

::: {#9322edbd-ba64-40a0-a138-2d84435e9fc3 .cell .code execution_count="95" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="9322edbd-ba64-40a0-a138-2d84435e9fc3" outputId="c2aeee30-cf41-4c5c-92db-0da85a0e5668"}
``` python
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    Index: 17265 entries, 0 to 49121
    Data columns (total 88 columns):
     #   Column                                                        Non-Null Count  Dtype  
    ---  ------                                                        --------------  -----  
     0   Age                                                           17265 non-null  int64  
     1   EdLevel                                                       17265 non-null  int64  
     2   Employment                                                    17265 non-null  int64  
     3   WorkExp                                                       17265 non-null  float64
     4   YearsCode                                                     17265 non-null  float64
     5   OrgSize                                                       17265 non-null  int64  
     6   RemoteWork                                                    17265 non-null  int64  
     7   ConvertedCompYearly                                           17265 non-null  float64
     8   Language_JavaScript                                           17265 non-null  int64  
     9   Language_HTML/CSS                                             17265 non-null  int64  
     10  Language_SQL                                                  17265 non-null  int64  
     11  Language_Python                                               17265 non-null  int64  
     12  Language_TypeScript                                           17265 non-null  int64  
     13  Language_Bash/Shell (all shells)                              17265 non-null  int64  
     14  Language_C#                                                   17265 non-null  int64  
     15  Language_Java                                                 17265 non-null  int64  
     16  Language_PowerShell                                           17265 non-null  int64  
     17  Language_C++                                                  17265 non-null  int64  
     18  Database_PostgreSQL                                           17265 non-null  int64  
     19  Database_MySQL                                                17265 non-null  int64  
     20  Database_SQLite                                               17265 non-null  int64  
     21  Database_Redis                                                17265 non-null  int64  
     22  Database_Microsoft SQL Server                                 17265 non-null  int64  
     23  Database_Unknown                                              17265 non-null  int64  
     24  Database_MongoDB                                              17265 non-null  int64  
     25  Database_MariaDB                                              17265 non-null  int64  
     26  Database_Elasticsearch                                        17265 non-null  int64  
     27  Database_Dynamodb                                             17265 non-null  int64  
     28  Platform_Docker                                               17265 non-null  int64  
     29  Platform_npm                                                  17265 non-null  int64  
     30  Platform_Amazon Web Services (AWS)                            17265 non-null  int64  
     31  Platform_Pip                                                  17265 non-null  int64  
     32  Platform_Kubernetes                                           17265 non-null  int64  
     33  Platform_Unknown                                              17265 non-null  int64  
     34  Platform_Homebrew                                             17265 non-null  int64  
     35  Platform_Vite                                                 17265 non-null  int64  
     36  Platform_Microsoft Azure                                      17265 non-null  int64  
     37  Platform_Google Cloud                                         17265 non-null  int64  
     38  Webframe_Node.js                                              17265 non-null  int64  
     39  Webframe_React                                                17265 non-null  int64  
     40  Webframe_Unknown                                              17265 non-null  int64  
     41  Webframe_jQuery                                               17265 non-null  int64  
     42  Webframe_ASP.NET Core                                         17265 non-null  int64  
     43  Webframe_Next.js                                              17265 non-null  int64  
     44  Webframe_Angular                                              17265 non-null  int64  
     45  Webframe_Express                                              17265 non-null  int64  
     46  Webframe_Vue.js                                               17265 non-null  int64  
     47  Webframe_ASP.NET                                              17265 non-null  int64  
     48  DevEnvs_Visual Studio Code                                    17265 non-null  int64  
     49  DevEnvs_Visual Studio                                         17265 non-null  int64  
     50  DevEnvs_IntelliJ IDEA                                         17265 non-null  int64  
     51  DevEnvs_Notepad++                                             17265 non-null  int64  
     52  DevEnvs_Unknown                                               17265 non-null  int64  
     53  DevEnvs_Vim                                                   17265 non-null  int64  
     54  DevEnvs_Cursor                                                17265 non-null  int64  
     55  DevEnvs_Android Studio                                        17265 non-null  int64  
     56  DevEnvs_PyCharm                                               17265 non-null  int64  
     57  DevEnvs_Neovim                                                17265 non-null  int64  
     58  DevType_Developer, full-stack                                 17265 non-null  int64  
     59  DevType_Developer, back-end                                   17265 non-null  int64  
     60  DevType_Architect, software or solutions                      17265 non-null  int64  
     61  DevType_Developer, desktop or enterprise applications         17265 non-null  int64  
     62  DevType_Developer, front-end                                  17265 non-null  int64  
     63  DevType_Developer, mobile                                     17265 non-null  int64  
     64  DevType_Developer, embedded applications or devices           17265 non-null  int64  
     65  DevType_Engineering manager                                   17265 non-null  int64  
     66  DevType_DevOps engineer or professional                       17265 non-null  int64  
     67  DevType_Data engineer                                         17265 non-null  int64  
     68  Country_Canada                                                17265 non-null  bool   
     69  Country_France                                                17265 non-null  bool   
     70  Country_Germany                                               17265 non-null  bool   
     71  Country_India                                                 17265 non-null  bool   
     72  Country_Netherlands                                           17265 non-null  bool   
     73  Country_Other                                                 17265 non-null  bool   
     74  Country_Poland                                                17265 non-null  bool   
     75  Country_Ukraine                                               17265 non-null  bool   
     76  Country_United Kingdom of Great Britain and Northern Ireland  17265 non-null  bool   
     77  Country_United States of America                              17265 non-null  bool   
     78  Industry_Fintech                                              17265 non-null  bool   
     79  Industry_Government                                           17265 non-null  bool   
     80  Industry_Healthcare                                           17265 non-null  bool   
     81  Industry_Internet, Telecomm or Information Services           17265 non-null  bool   
     82  Industry_Manufacturing                                        17265 non-null  bool   
     83  Industry_Other                                                17265 non-null  bool   
     84  Industry_Other:                                               17265 non-null  bool   
     85  Industry_Retail and Consumer Services                         17265 non-null  bool   
     86  Industry_Software Development                                 17265 non-null  bool   
     87  Industry_Transportation, or Supply Chain                      17265 non-null  bool   
    dtypes: bool(20), float64(3), int64(65)
    memory usage: 9.4 MB
:::
:::

::: {#2e709b41-f28c-42c4-bb90-a6883b8ac2e4 .cell .code execution_count="96" id="2e709b41-f28c-42c4-bb90-a6883b8ac2e4"}
``` python
data = data.drop(columns=['Industry_Other:','DevEnvs_Unknown','Database_Unknown','Platform_Unknown',
                          'Webframe_Unknown'])
```
:::

::: {#10219ae2-af29-4748-9467-6df0b594e728 .cell .code execution_count="97" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":307}" id="10219ae2-af29-4748-9467-6df0b594e728" outputId="78186f97-5576-414e-da33-dce703220bb2"}
``` python
data.head()
```

::: {.output .execute_result execution_count="97"}
``` json
{"type":"dataframe","variable_name":"data"}
```
:::
:::

::: {#mbIFJ_A-mHbk .cell .markdown id="mbIFJ_A-mHbk"}
#### deleting outliers
:::

::: {#D1NqIPQfdcCP .cell .code execution_count="98" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="D1NqIPQfdcCP" outputId="b2747f0d-6c37-43a1-b465-cc066ff57fa2"}
``` python
Q1 = data['ConvertedCompYearly'].quantile(0.25)
Q3 = data['ConvertedCompYearly'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

print(f'Removing rows with salary > {upper} and < {lower}')

data = data[data['ConvertedCompYearly']<=upper]
data = data[data['ConvertedCompYearly']>=lower]
```

::: {.output .stream .stdout}
    Removing rows with salary > 252352.5 and < -84587.5
:::
:::

::: {#87fb81eb-3c25-4255-bcc3-0fdc5180c3f6 .cell .markdown id="87fb81eb-3c25-4255-bcc3-0fdc5180c3f6"}
# EDA
:::

::: {#edRmaX6QmsRM .cell .markdown id="edRmaX6QmsRM"}
#### Salary distribution
:::

::: {#2fc136c0-37af-4813-bd8e-5848085d410b .cell .code execution_count="99" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="2fc136c0-37af-4813-bd8e-5848085d410b" outputId="c1675680-e4b7-4297-ae09-2384e472563b"}
``` python
plt.figure(figsize=(10,5))
plt.hist(data['ConvertedCompYearly'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram - yearly salary')
plt.xlabel('Yearly salary ($)')
plt.show()
```

::: {.output .display_data}
![](35d77ac1339e34e311cfcb58cea052477bbcd3b8.png)
:::
:::

::: {#3KJjaUa-fjpI .cell .markdown id="3KJjaUa-fjpI"}
**Conclusions**

-   **right-skwed distribiution: most people earn low salary, only a
    small number have high income**
-   **as salary increases, the number of people decreses**
-   **most common salaries are in between 45-80k \$**
:::

::: {#rdwZ0tMEm8nh .cell .markdown id="rdwZ0tMEm8nh"}
#### Salary vs Age
:::

::: {#9NQo906c9IA- .cell .code execution_count="100" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="9NQo906c9IA-" outputId="a19a75cd-95e6-4681-c9d2-3506be96e389"}
``` python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
df_grouped = org_data.groupby('Age', as_index=False)['ConvertedCompYearly'].median()
sns.barplot(df_grouped,x='Age', y ='ConvertedCompYearly', hue='ConvertedCompYearly',palette='flare',legend=False)
plt.xlabel('Age groups')
plt.xticks(rotation=30)
plt.ylabel('Annual salary ($)')
plt.title('Salary vs Age')

plt.subplot(1,2,2)
plt.hist(org_data['Age'])
plt.xticks(rotation=30)
plt.title('Age histogram')
plt.ylabel('Count')
plt.xlabel('Age')
plt.show()
```

::: {.output .display_data}
![](f902771bcb3b514c668611159ece89cf4c99eb84.png)
:::
:::

::: {#6lu33U6Bjpdj .cell .markdown id="6lu33U6Bjpdj"}
**Conclusions**

-   **clear upward trend (strong positive correlation): salary increses
    with age**
-   **the highest difference is between age groups: 18-24 and 25-34
    years old due to career progression**
-   **between age groups 55-64 and 65+ years old trend starts to
    flatten**
-   **the age distribution is right-skewed (more data in younger/middle
    ages, fewer in older ages). The dataset is dominated by people aged
    25--34 and 35--44**
:::

::: {#mjoLwQNMnC9R .cell .markdown id="mjoLwQNMnC9R"}
#### Top Developer Types by Salary
:::

::: {#PofYZfkJFLAL .cell .code execution_count="101" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="PofYZfkJFLAL" outputId="b5727c0a-9e7c-4f1e-ff26-abedfca8c646"}
``` python
plt.figure(figsize=(15,5))
org_data2 = org_data.copy()
org_data2['DevType'] = org_data['DevType'].apply(lambda x: " ".join(x))
df_grouped2 = org_data2.groupby('DevType', as_index=False)['ConvertedCompYearly'].median().sort_values(by='ConvertedCompYearly',ascending=False).head(10)
sns.barplot(df_grouped2,y='DevType', x ='ConvertedCompYearly', hue='ConvertedCompYearly',palette='flare',legend=False,orient='h')
plt.ylabel('DevType')
plt.xlabel('Yearly salary')
plt.title('Top 10 best DevType by Salary')
plt.show()
```

::: {.output .display_data}
![](3a814716a1570ca79945e4819d95bf6923422cfd.png)
:::
:::

::: {#cmljKV_glh_R .cell .markdown id="cmljKV_glh_R"}
**Conclusions**

-   **highest-paying role are:**

1.  Financial analyst or engineer (\~145k \$)

2.  Engineering manager (\~135k \$)

3.  Senior executive (\~130k \$)

**these roles require technical expertise combined with strong
leadership skills**

-   **other high-paying roles (cloud, security, architecture) require
    specialized technical skills and expertise as well**
-   **the salaries for the other roles are quite similar to each other,
    mostly falling within the range of 90k to 110k \$**
:::

::: {#vkv3mgrlnH6w .cell .markdown id="vkv3mgrlnH6w"}
#### Top Countries by Salary
:::

::: {#cXTh6ERyY1oJ .cell .code execution_count="102" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="cXTh6ERyY1oJ" outputId="026ea286-ddc4-4935-d573-49f2b9378a3c"}
``` python
val_counts = org_data['Country'].value_counts()
countries_temp = val_counts[val_counts>15].index # countries that accoured at least 16 times

plt.figure(figsize=(15,5))
df_grouped = org_data[org_data['Country'].isin(countries_temp)].groupby('Country', as_index=False)['ConvertedCompYearly'].median().sort_values(by='ConvertedCompYearly',ascending=False).head(10)
sns.barplot(df_grouped,y='Country', x ='ConvertedCompYearly', hue='ConvertedCompYearly',palette='Spectral',legend=False,orient='h')
plt.ylabel('Country')
plt.xticks(rotation=40,fontsize=10)
plt.xlabel('Annual salary ($)')
plt.title('Top 10 best Countries by Salary')
plt.show()
```

::: {.output .display_data}
![](f3d27fd1bb8f3a28966086919c6547b905d26604.png)
:::
:::

::: {#z8OcGalXxwh8 .cell .markdown id="z8OcGalXxwh8"}
**Conclusions**

-   **the highest salary are observed in United States, Switzerland and
    Israel, at around 140k \$**

-   **there is a visible gap between top 3 best countries and the rest**

-   **salaries in the remaining countries are relatively similar,
    generally ranging from 90k to 100k \$**

-   **most of the countries in the ranking are highly developed
    economies**
:::

::: {#KKMy-VMlYEiY .cell .markdown id="KKMy-VMlYEiY"}
#### Salary vs Work Experience
:::

::: {#mDcuKP2_r5wj .cell .code execution_count="103" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="mDcuKP2_r5wj" outputId="b50d251e-aad0-495a-9901-70e6d8261598"}
``` python
plt.figure(figsize=(15,5))
filtered = org_data.groupby('WorkExp', as_index=False)['ConvertedCompYearly'].median()
org_data = org_data[org_data['ConvertedCompYearly']<320000]
org_data = org_data[org_data['WorkExp']<60]
sns.scatterplot(data=org_data, x='WorkExp', y='ConvertedCompYearly',alpha=0.3)
#sns.regplot(data=org_data, x='WorkExp',y='ConvertedCompYearly',color='r',scatter=False)
plt.plot(filtered['WorkExp'], filtered['ConvertedCompYearly'], color='r')
plt.xlabel('Work experience')
plt.ylabel('Annual Salary ($)')
plt.title('Salary vs Work experience')
plt.show()
```

::: {.output .display_data}
![](9ff998ba855c2fc5dc0d7cdfe614ad848044267e.png)
:::
:::

::: {#UVpv26ed16HZ .cell .markdown id="UVpv26ed16HZ"}
**Conclusions**

-   **there is a positive relationship between work experience and
    salary, especially during the first 10 years**
-   **between 1st and 10th year median salary tripples, reflecting
    carrer growth**
-   **after 10 years, upward trend continues, but becomes more gradual**
-   **after around 25 years, salaries tend to stabilize, with some
    fluctuations**
-   **beyond 45 years salary is irregular due to fewer observations and
    possible data inconsistencies**
:::

::: {#jiCA_H7vw2e0 .cell .code execution_count="104" id="jiCA_H7vw2e0"}
``` python
org_data['EdLevel'] = org_data['EdLevel'].replace('Master’s degree (M.A., M.S., M.Eng., MBA, etc.)','Master’s degree')
org_data['EdLevel'] = org_data['EdLevel'].replace('Bachelor’s degree (B.A., B.S., B.Eng., etc.)','Bachelor’s degree')
org_data['EdLevel'] = org_data['EdLevel'].replace('Professional degree (JD, MD, Ph.D, Ed.D, etc.)','Professional degree')
org_data['EdLevel'] = org_data['EdLevel'].replace('Other (please specify):','Other')
org_data['EdLevel'] = org_data['EdLevel'].replace('Some college/university study without earning a degree','College study without earning a degree')
org_data['EdLevel'] = org_data['EdLevel'].replace('Associate degree (A.A., A.S., etc.)','Associate degree')
org_data['EdLevel'] = org_data['EdLevel'].replace('Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
                                                  'Secondary school')
order_edu = [
    'Primary/elementary school',
    'Secondary school',
    'College study without earning a degree',
    'Associate degree',
    'Bachelor’s degree',
    'Master’s degree',
    'Professional degree',
    'Other']
```
:::

::: {#3DRwevBxX_Ra .cell .markdown id="3DRwevBxX_Ra"}
#### Salary vs Education
:::

::: {#FSKrw0T_yvUo .cell .code execution_count="105" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="FSKrw0T_yvUo" outputId="337f11c3-0a82-433e-d498-937798ff9b84"}
``` python
plt.figure(figsize=(15,5))
filtered = org_data.groupby('EdLevel', as_index=False)['ConvertedCompYearly'].median()
sns.barplot(data=filtered, x='EdLevel', y='ConvertedCompYearly',order=order_edu,hue='ConvertedCompYearly',legend=False)
plt.xticks(rotation=30)
plt.xlabel('Education level')
plt.ylabel('Annual salary ($)')
plt.title('Salary by education level')
plt.show()
```

::: {.output .display_data}
![](07431e439acdcace68eaa20a2c4abac58349a996.png)
:::
:::

::: {#e18TuD2nhHcY .cell .code execution_count="106" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="e18TuD2nhHcY" outputId="4c8f8325-eeff-415f-a42c-82cf686206e2"}
``` python
filtered
```

::: {.output .execute_result execution_count="106"}
``` json
{"summary":"{\n  \"name\": \"filtered\",\n  \"rows\": 8,\n  \"fields\": [\n    {\n      \"column\": \"EdLevel\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 8,\n        \"samples\": [\n          \"Bachelor\\u2019s degree\",\n          \"Primary/elementary school\",\n          \"Associate degree\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"ConvertedCompYearly\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 11478.25468549117,\n        \"min\": 52530.0,\n        \"max\": 87462.0,\n        \"num_unique_values\": 8,\n        \"samples\": [\n          76570.0,\n          65268.5,\n          71749.5\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe","variable_name":"filtered"}
```
:::
:::

::: {#o0YznEC66qWe .cell .markdown id="o0YznEC66qWe"}
**Conclusions**

-   **there is a positive relationship between education level and
    salary**
-   **the data suggests that higher education leads to higher salaries**
-   **the highest salaries are observed for professional and master\'s
    degrees**
-   **people with a Bachelor\'s degree earn about 18k more than those
    with secondary school education, while those with a Professional
    degree earn almost \$30k more**
:::

::: {#14611c84-756e-42f3-b77b-6c79749b57e2 .cell .code execution_count="107" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="14611c84-756e-42f3-b77b-6c79749b57e2" outputId="55852c60-3d33-4b96-ac9f-3e12ac485069"}
``` python
data['ConvertedCompYearly'].describe()
```

::: {.output .execute_result execution_count="107"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ConvertedCompYearly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16610.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>83859.114389</td>
    </tr>
    <tr>
      <th>std</th>
      <td>58245.329899</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40000.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75410.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>118593.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>252000.000000</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> float64</label>
```
:::
:::

::: {#4c674674-e660-438a-b72e-489ab2300c9b .cell .code execution_count="108" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4c674674-e660-438a-b72e-489ab2300c9b" outputId="8d23d8f4-7fe5-4cd0-b86c-1d6111314ec2"}
``` python
data.columns[data.isna().any()]
```

::: {.output .execute_result execution_count="108"}
    Index([], dtype='object')
:::
:::

::: {#81e52fb5-3fdf-4f0c-a316-bc72c5a0ca6d .cell .code execution_count="109" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="81e52fb5-3fdf-4f0c-a316-bc72c5a0ca6d" outputId="1afdf57d-404a-4354-ec69-b4e55b42a2ce"}
``` python
print(f'Number of rows: {data.shape[0]} \nNumber of columns : {data.shape[1]}')
```

::: {.output .stream .stdout}
    Number of rows: 16610 
    Number of columns : 83
:::
:::

::: {#ff4d572f-d1e0-4256-8fba-61c59db28449 .cell .markdown id="ff4d572f-d1e0-4256-8fba-61c59db28449"}
#### Variables most correlated with salary
:::

::: {#5dda12ee-132a-423d-a8e9-00397d9f2584 .cell .code execution_count="110" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="5dda12ee-132a-423d-a8e9-00397d9f2584" outputId="c10b0ce3-41dd-41b4-b47e-16c7294ff579"}
``` python
cor = data.drop(columns='ConvertedCompYearly').corrwith(data['ConvertedCompYearly'])
print(cor.sort_values(ascending=True).tail(10).plot(kind='barh'))
plt.title('Top positive correlation')
plt.xlabel('Correlation')
```

::: {.output .stream .stdout}
    Axes(0.125,0.11;0.775x0.77)
:::

::: {.output .execute_result execution_count="110"}
    Text(0.5, 0, 'Correlation')
:::

::: {.output .display_data}
![](4978191997a75ad914077bc44155e06b66fb079b.png)
:::
:::

::: {#vacsPqBGk1aL .cell .markdown id="vacsPqBGk1aL"}
-   **being located in the United States shows the strongest positive
    correlation with salary (\~0.5)**
-   **years of coding experience (YearsCode) and work experience
    (WorkExp) are higly correlated (\~0.39 and \~0.36), suggesting that
    more experience leads to higher salaries. However, these two
    variables might be higly correlated, which has to be checked to
    avoid multicollinearity in the model**
-   **remote work, organization size and the other variables have lower
    correlations, indicating some impact but less than experience or
    location in the USA**
:::

::: {#8fdd1550-70e6-433a-855b-8cde719e632d .cell .code execution_count="111" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":0}" id="8fdd1550-70e6-433a-855b-8cde719e632d" outputId="be61732a-c3e5-46ba-c2f9-d51d176b939d"}
``` python
print(cor.sort_values(ascending=True).head(10).plot(kind='barh'))
plt.title('Top negative correlation')
plt.xlabel('Correlation')
```

::: {.output .stream .stdout}
    Axes(0.125,0.11;0.775x0.77)
:::

::: {.output .execute_result execution_count="111"}
    Text(0.5, 0, 'Correlation')
:::

::: {.output .display_data}
![](b28eb7f2e22203a79bbb48119dbc0ec6c92cda45.png)
:::
:::

::: {#MA19oiKWmrf8 .cell .markdown id="MA19oiKWmrf8"}
-   **not being located in the most common countries shows the strongest
    negative correlation with the salary**
-   **working in India or Ukraine are another negatively correlated
    features**
-   **other variables show relatively low correlations, which suggests
    they have less consistent impact on salary compared to mentioned
    above features**
:::

::: {#19cf903f-a6c4-4928-b6e5-dced7398aa60 .cell .code execution_count="112" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="19cf903f-a6c4-4928-b6e5-dced7398aa60" outputId="245b6215-d9fd-452b-fcfa-794cff726dc0"}
``` python
data['WorkExp'].corr(data['YearsCode'])
# due to high correlation between Work Experience and YearsCode i'm deleting one of the columns to avoid multicorrelation
```

::: {.output .execute_result execution_count="112"}
    np.float64(0.8854387184199839)
:::
:::

::: {#8ca328b9-9786-4266-b8bb-ca0d1e2780c0 .cell .code execution_count="113" id="8ca328b9-9786-4266-b8bb-ca0d1e2780c0"}
``` python
data = data.drop(columns='YearsCode')
```
:::

::: {#SsFj2SK-mqBC .cell .markdown id="SsFj2SK-mqBC"}
# Modeling
:::

::: {#b7D2nxcFmtSR .cell .markdown id="b7D2nxcFmtSR"}
## Baseline model
:::

::: {#a6ySqS4Avk_Y .cell .code execution_count="114" id="a6ySqS4Avk_Y"}
``` python
from sklearn.dummy import DummyRegressor
```
:::

::: {#f3bf30ad-4a20-4e66-9c07-c99b7bcf7d21 .cell .code execution_count="115" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="f3bf30ad-4a20-4e66-9c07-c99b7bcf7d21" outputId="450210cf-ff48-4f38-a6fd-56045372bbbe"}
``` python
X = data.drop(columns='ConvertedCompYearly')
y = data['ConvertedCompYearly']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

print(f'Size of the training set: {X_train.shape}')
print(f'Size of the testing set: {X_test.shape}')
```

::: {.output .stream .stdout}
    Size of the training set: (13288, 81)
    Size of the testing set: (3322, 81)
:::
:::

::: {#xLPx7_R1j0AG .cell .code execution_count="116" id="xLPx7_R1j0AG"}
``` python
results = pd.DataFrame(columns=['Model','R2 train','R2 test', 'MAE test','RMSE test'])
```
:::

::: {#DyktvNw1lRzL .cell .code execution_count="117" id="DyktvNw1lRzL"}
``` python
def build_model(model,model_name, X_train, X_test, y_train, y_test,chart=0):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    evaluation_model(model,model_name,X_train,X_test,y_train,y_test,y_pred_train,y_pred_test)
    if chart:
        actual_predicted_plot(y_test, y_pred_test)
    #return model
```
:::

::: {#BAGPdjwDiUyY .cell .code execution_count="118" id="BAGPdjwDiUyY"}
``` python
def evaluation_model(model,model_name,X_train,X_test,y_train,y_test,y_pred_train,y_pred_test):
  r2_train = r2_score(y_train, y_pred_train)
  r2_test = r2_score(y_test, y_pred_test)
  mae_train = mean_absolute_error(y_train, y_pred_train)
  mae_test = mean_absolute_error(y_test, y_pred_test)
  rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
  print(f'--- Model -> {model_name}---')
  print(f'Evaluation - Training set')
  print(f'R2 score : {r2_train:.2f}')
  print(f'MAE score : {mae_train:,.2f} $')
  print(f'\nEvaluation - Test set')
  print(f'R2 score : {r2_test:.2f}')
  print(f'MAE score : {mae_test:,.2f} $')
  results.loc[len(results)] = {'Model':model_name,'R2 train': np.round(r2_train,2),
                               'R2 test':np.round(r2_test,2), 'MAE test':np.round(mae_test,2),
                               'RMSE test': np.round(rmse_test,2)}
  return results
```
:::

::: {#ZcL8UduTlxwn .cell .code execution_count="119" id="ZcL8UduTlxwn"}
``` python
def actual_predicted_plot(y_test, y_pred_test):
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  plt.scatter(y_test, y_pred_test, alpha=0.5)
  plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--' )
  plt.xlabel('Actual salary ($)')
  plt.ylabel('Predicted salary ($)')
  plt.title('Actual vs Predicted salary')
  plt.subplot(1,2,2)
  residuals = y_test - y_pred_test
  plt.scatter(y_pred_test, residuals,alpha=0.5)
  plt.axhline(y=0,color='r')
  plt.title('Residual plot')
  plt.xlabel('Predicted salary ($)')
  plt.ylabel('Residuals ($)')
```
:::

::: {#aGpUs_egvnBz .cell .code execution_count="120" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="aGpUs_egvnBz" outputId="d064ebc8-eb7b-4d5b-b5f3-c380cc3f7776"}
``` python
baseline = DummyRegressor(strategy='median')
build_model(baseline,'Baseline', X_train, X_test, y_train, y_test)
```

::: {.output .stream .stdout}
    --- Model -> Baseline---
    Evaluation - Training set
    R2 score : -0.02
    MAE score : 46,225.59 $

    Evaluation - Test set
    R2 score : -0.03
    MAE score : 46,954.58 $
:::
:::

::: {#f790e5aa-f0c9-4ef6-a505-04b7ff45269f .cell .markdown id="f790e5aa-f0c9-4ef6-a505-04b7ff45269f"}
## Linear Regression
:::

::: {#fac2c98a-40d1-43a4-96e8-858b13d26bf2 .cell .code execution_count="121" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="fac2c98a-40d1-43a4-96e8-858b13d26bf2" outputId="62f29dd0-e067-4cba-a41d-3818da046d1b"}
``` python
model = LinearRegression()
build_model(model,'Linear Regression', X_train, X_test, y_train, y_test)
```

::: {.output .stream .stdout}
    --- Model -> Linear Regression---
    Evaluation - Training set
    R2 score : 0.52
    MAE score : 29,814.24 $

    Evaluation - Test set
    R2 score : 0.53
    MAE score : 30,022.73 $
:::
:::

::: {#959cbc28-804b-4b74-be89-9b1969e0334d .cell .markdown id="959cbc28-804b-4b74-be89-9b1969e0334d"}
#### Top highest and lowest coefficients
:::

::: {#9aebe6c9-d0ca-4fb2-a8e2-70d84855b95d .cell .code execution_count="122" id="9aebe6c9-d0ca-4fb2-a8e2-70d84855b95d"}
``` python
df_coef = pd.DataFrame(np.round(model.coef_,2),index= X.columns, columns=['Coef'])
```
:::

::: {#eacd63b9-fa53-4030-ba90-3c3a156a1b64 .cell .code execution_count="123" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":363}" id="eacd63b9-fa53-4030-ba90-3c3a156a1b64" outputId="4c752287-874b-4c99-ebdb-19370e4b8079"}
``` python
df_coef.sort_values(by='Coef', ascending=False).head(10)
```

::: {.output .execute_result execution_count="123"}
``` json
{"summary":"{\n  \"name\": \"df_coef\",\n  \"rows\": 10,\n  \"fields\": [\n    {\n      \"column\": \"Coef\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 23910.3428010121,\n        \"min\": 9163.81,\n        \"max\": 94311.23,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          25402.36,\n          57872.52,\n          28456.62\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {#2C9n-DwuCYJp .cell .markdown id="2C9n-DwuCYJp"}
-   **location has a very stronger effect on the salary, with the USA
    having the highest positive coefficient (94k \$)**

-   **the second top-tier countries for salary are Canada, Netherlands
    and Germany (42-57k \$)**

-   **engineering manager roles show the strongest positive influence on
    salary**
:::

::: {#78a031fa-2738-4f91-8e59-34f80168443e .cell .code execution_count="124" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":363}" id="78a031fa-2738-4f91-8e59-34f80168443e" outputId="3bc01ec2-0305-452c-be72-f20bc8287a60"}
``` python
df_coef.sort_values(by='Coef', ascending=False).tail(10)
```

::: {.output .execute_result execution_count="124"}
``` json
{"summary":"{\n  \"name\": \"df_coef\",\n  \"rows\": 10,\n  \"fields\": [\n    {\n      \"column\": \"Coef\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1125.5895194810387,\n        \"min\": -7725.54,\n        \"max\": -3620.72,\n        \"num_unique_values\": 10,\n        \"samples\": [\n          -5544.15,\n          -3880.75,\n          -5157.3\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {#OmepbFB-cSFI .cell .markdown id="OmepbFB-cSFI"}
-   **several technologies (such as MySQL, jQuery, Express, HTML/CSS)
    have negative coefficients, suggesting they are more common in
    lower-paying roles**
-   **some tools and environments (such as Android Studio, MongoDB,
    MariaDB) are also negatively associated with salary, possibly due to
    the types of roles or companies where they are used**
-   **manufacturing industry is associated with lower salaries**
-   **the negative impact of age on salary might be due to higher
    proportion of younger respondents in the data, who typically earn
    less early in their careers**
:::

::: {#dSQsyLUh2Ldp .cell .markdown id="dSQsyLUh2Ldp"}
## Gradient Boosting
:::

::: {#BneKm-fQ2Q7G .cell .code execution_count="125" id="BneKm-fQ2Q7G"}
``` python
from sklearn.ensemble import GradientBoostingRegressor
```
:::

::: {#805fgAyQ2ST2 .cell .code execution_count="126" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="805fgAyQ2ST2" outputId="794b621b-6684-441e-8602-9e6e46920596"}
``` python
gbr = GradientBoostingRegressor(n_estimators=100)
build_model(gbr,'Gradient Boost Regressor', X_train, X_test, y_train, y_test,chart=0)
```

::: {.output .stream .stdout}
    --- Model -> Gradient Boost Regressor---
    Evaluation - Training set
    R2 score : 0.55
    MAE score : 28,688.08 $

    Evaluation - Test set
    R2 score : 0.54
    MAE score : 29,483.97 $
:::
:::

::: {#BpPIcBh-31Pl .cell .markdown id="BpPIcBh-31Pl"}
### Grid Search for Gradient Boosting
:::

::: {#80dEoBGa30eu .cell .code execution_count="127" id="80dEoBGa30eu"}
``` python
from sklearn.model_selection import GridSearchCV
```
:::

::: {#4y5LaIQH37Fs .cell .code execution_count="128" id="4y5LaIQH37Fs"}
``` python
param = {'n_estimators':[250,500],
         'learning_rate': [0.05,0.1],
         'max_depth': [2,3,5]}
```
:::

::: {#chgy-9yIUhZ2 .cell .code execution_count="129" id="chgy-9yIUhZ2"}
``` python
def grid_search(model,param):
    grid = GridSearchCV(estimator = model, param_grid=param,cv=3)
    grid.fit(X_train,y_train)
    print(f'Best score: {grid.best_score_} using parameters: {grid.best_params_}')
    return grid
```
:::

::: {#2IQFQUEXU2Xf .cell .code execution_count="130" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2IQFQUEXU2Xf" outputId="62125104-00fe-4e63-fa33-88f9ec856372"}
``` python
grid1 = grid_search(gbr,param)
```

::: {.output .stream .stdout}
    Best score: 0.5402316738560883 using parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 500}
:::
:::

::: {#i8vYudbjuYFU .cell .markdown id="i8vYudbjuYFU"}
### Gradient Boosting with the best parameters
:::

::: {#jpChdhDoxVTE .cell .code execution_count="131" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":534}" id="jpChdhDoxVTE" outputId="9e2f603e-0ba3-4cfe-cb4f-909ad6a4f37c"}
``` python
model = GradientBoostingRegressor(**grid1.best_params_)
build_model(model,'GB Regressor + GridSearch', X_train, X_test, y_train, y_test,chart=1)
```

::: {.output .stream .stdout}
    --- Model -> GB Regressor + GridSearch---
    Evaluation - Training set
    R2 score : 0.61
    MAE score : 26,533.14 $

    Evaluation - Test set
    R2 score : 0.56
    MAE score : 28,679.38 $
:::

::: {.output .display_data}
![](b10697ebd51489dfe8db75ed8fc83f92dcf08fe4.png)
:::
:::

::: {#OL6SWsqGPjpj .cell .markdown id="OL6SWsqGPjpj"}
-   **the model captures the general trend, but struggles to accurately
    predict medium (\~\$150k) and high salaries**
-   **as salary increases, residuals decrease and become more negative,
    indicating growing error and heteroscedasticity**
-   **for higher salaries the model tends to underestimate, while for
    lower salaries it often overestimates**
:::

::: {#T0tl4z7T7ofP .cell .markdown id="T0tl4z7T7ofP"}
## Random Forest + GridSearch {#random-forest--gridsearch}
:::

::: {#JQAZZnNP7j1n .cell .code execution_count="132" id="JQAZZnNP7j1n"}
``` python
from sklearn.ensemble import RandomForestRegressor
```
:::

::: {#OTNm_U0g8A6V .cell .code execution_count="133" id="OTNm_U0g8A6V"}
``` python
param2 = {'n_estimators':[100,200],
         'min_samples_leaf': [1, 2],
         'max_depth': [None,20]}
```
:::

::: {#rQhE6icg7mD1 .cell .code execution_count="134" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="rQhE6icg7mD1" outputId="bcd5e01f-4ae8-4df6-ec61-b6a22eeb6812"}
``` python
model = RandomForestRegressor()
grid_r = grid_search(model, param2)
model = RandomForestRegressor(**grid_r.best_params_)
build_model(model,'Random Forest + GridSearch', X_train, X_test, y_train, y_test,chart=0)
```

::: {.output .stream .stdout}
    Best score: 0.5187990844031293 using parameters: {'max_depth': None, 'min_samples_leaf': 2, 'n_estimators': 200}
    --- Model -> Random Forest + GridSearch---
    Evaluation - Training set
    R2 score : 0.89
    MAE score : 13,199.95 $

    Evaluation - Test set
    R2 score : 0.54
    MAE score : 29,415.59 $
:::
:::

::: {#tEa1TjTEUSST .cell .markdown id="tEa1TjTEUSST"}
## Gradient Boosting - GridSearch + Feature Importance {#gradient-boosting---gridsearch--feature-importance}
:::

::: {#9MtfG_6LUfjZ .cell .code execution_count="135" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":489}" id="9MtfG_6LUfjZ" outputId="fdae6068-58d0-4817-98ff-fcad04f6b977"}
``` python
feature_importance = gbr.feature_importances_
feature_importance_df = pd.DataFrame({'Feature':X.columns, 'Importance':gbr.feature_importances_})
top20 = feature_importance_df.sort_values(by='Importance', ascending=False).head(20)
sns.barplot(top20, x='Importance', y='Feature')
plt.title('Top 20 most important features')
```

::: {.output .execute_result execution_count="135"}
    Text(0.5, 1.0, 'Top 20 most important features')
:::

::: {.output .display_data}
![](e260a36d872d461734c82e2d307bcc3d5f02b099.png)
:::
:::

::: {#Qru4fX5z-DsJ .cell .markdown id="Qru4fX5z-DsJ"}
-   **7 out of the top 20 features are location-based, indicating a
    strong geographic influence on salary**
-   **the most impactful factors are being located in the USA and work
    experience**
-   **other features (such as organisation size, platforms, employment,
    age) have significantly lower impact**
:::

::: {#P0OO0HFeZ5vN .cell .code execution_count="136" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="P0OO0HFeZ5vN" outputId="71457653-1bf3-4187-9c8f-d6a9ed4d3a4f"}
``` python
data_top = data[top20['Feature']]
X_train,X_test,y_train,y_test = train_test_split(data_top,y,test_size=0.2, random_state=42)

gbr2= GradientBoostingRegressor()
gbr2.fit(X_train, y_train)
grid2 = grid_search(gbr2,param)
```

::: {.output .stream .stdout}
    Best score: 0.5227709641968801 using parameters: {'learning_rate': 0.1, 'max_depth': 2, 'n_estimators': 500}
:::
:::

::: {#bLM7Nxwno-LN .cell .code execution_count="137" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="bLM7Nxwno-LN" outputId="59e33085-3c84-4f79-abb0-3981648eee0f"}
``` python
model = GradientBoostingRegressor(**grid2.best_params_)
build_model(model,'GB Regressor + GridSearch + FeatureImportance', X_train, X_test, y_train, y_test,chart=0)
```

::: {.output .stream .stdout}
    --- Model -> GB Regressor + GridSearch + FeatureImportance---
    Evaluation - Training set
    R2 score : 0.54
    MAE score : 28,980.78 $

    Evaluation - Test set
    R2 score : 0.55
    MAE score : 29,331.17 $
:::
:::

::: {#yGNOxz8Sql6H .cell .markdown id="yGNOxz8Sql6H"}
# Evaluation
:::

::: {#8pviF9VX1qlw .cell .code execution_count="138" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":237}" id="8pviF9VX1qlw" outputId="cb47d223-0d02-4bbf-c264-236d8409237c"}
``` python
results.sort_values(by='Model')
```

::: {.output .execute_result execution_count="138"}
``` json
{"summary":"{\n  \"name\": \"results\",\n  \"rows\": 6,\n  \"fields\": [\n    {\n      \"column\": \"Model\",\n      \"properties\": {\n        \"dtype\": \"string\",\n        \"num_unique_values\": 6,\n        \"samples\": [\n          \"Baseline\",\n          \"GB Regressor + GridSearch\",\n          \"Random Forest + GridSearch\"\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"R2 train\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.29588849251026983,\n        \"min\": -0.02,\n        \"max\": 0.89,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          -0.02,\n          0.61,\n          0.89\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"R2 test\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0.23455631875237698,\n        \"min\": -0.03,\n        \"max\": 0.56,\n        \"num_unique_values\": 5,\n        \"samples\": [\n          0.56,\n          0.53,\n          0.55\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"MAE test\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 7184.902040838136,\n        \"min\": 28679.38,\n        \"max\": 46954.58,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          46954.58,\n          28679.38,\n          29415.59\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"RMSE test\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 8218.527054096738,\n        \"min\": 38933.71,\n        \"max\": 59808.57,\n        \"num_unique_values\": 6,\n        \"samples\": [\n          59808.57,\n          38933.71,\n          39887.86\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}","type":"dataframe"}
```
:::
:::

::: {#89voENo0xKdT .cell .markdown id="89voENo0xKdT"}
-   **The best performing model was the Gradient Boost Regressor with
    Grid Search, as it achieved the highest R2 scores and the lowest
    errors (MAE \~29k \$)**

-   **Compared to the baseline model (MAE \~47k \$), it achieved
    significantly better accuracy in predicting salary**

-   **Linear Regression showed the weakest performance among the tested
    models (excluding the baseline), suggesting that the relationships
    in the data are not linear**

-   **Feature importance slightly worsens the Gradient Boost Regressor
    results, while Grid Search improves them**

-   **Random Forest shows signs of overfitting (high R2 score on train
    data and lower on test data), which suggests that the model has
    memorized the training data and does not generalize well to unseen
    observations**
:::

::: {#YSlxPvF9yh07 .cell .markdown id="YSlxPvF9yh07"}
# Conclusions
:::

::: {#aHkbiXImyk4e .cell .markdown id="aHkbiXImyk4e"}
-   **the analysis showed that the most important factors influencing
    salary are location (particularly being based in the United States)
    and years of professional experience. Other variables, such as
    company size, have a noticeably smaller impact on salary levels**
-   **among the tested models, the best performance was achieved by the
    Gradient Boosting Regressor combined with Grid Search, which
    resulted in the highest predictive accuracy and the lowest error
    metrics**
-   **other models had worse performence due to non-linear data (Linear
    Regression) or overfitting - achieving better results on trained
    than unseen data (Random Forest)**
-   **limitations: noisy data, the data is dominated by respondents with
    lower to mid-level salaries, resulting in a skewed salary
    distribution and limited representation of high earners. This makes
    accurate prediction of extreme salary values more difficult**
-   **despite limitations, the models significantly outperform the
    baseline approach, indicating that salary can be predicted using the
    available features**
:::

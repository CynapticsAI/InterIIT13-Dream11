// const BASE_URL = "http://127.0.0.1:8000/backend" + "/api/v1";
const BASE_URL = "https://qwertyweb.xyz:8443/backend" + "/api/v1";

async function apiCall(endpoint, params = {}, token = null, options = {}) {
  const url = `${BASE_URL}/${endpoint}`;
  
  const queryParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      queryParams.append(key, value);
    }
  });

  const fullUrl = `${url}${
    queryParams.toString() ? `?${queryParams.toString()}` : ""
  }`;

  const headers = {
    "Content-Type": "application/json",
    ...(token && { Authorization: `Bearer ${token}` }),
    ...options.headers,
  };

  const requestOptions = {
    cache: "no-store",
    headers,
    ...options,
  };

  try {
    console.log(fullUrl)
    const response = await fetch(fullUrl, requestOptions);

    if (response.status === 401) {
      throw new Error("Unauthorized");
    }

    return await response.json();
  } catch (error) {
    console.error(`Error fetching data from ${fullUrl}:`, error);
    throw error;
  }
}

export function getMatches(params = {}, token = null, options = {}) {
  return apiCall("matches", params, token, options);
}

export function getAllPlayers(params = {}, token = null, options = {}) {
  return apiCall("players", params, token, options);
}

export function getAllTeams(params = {}, token = null, options = {}) {
  return apiCall("teams", params, token, options);
}

export function getPlayerData(params = {}, token = null, options = {}) {
  return apiCall(`get-player-report/${params.player_identifier}`,{}, token, options);
}

export function getMatchRelatedChats(params = {}, token = null, options = {}) {
  console.log(params)
  return apiCall("get-match-related-chats", params, token, options);
}


